import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import torchvision.transforms as transforms


def calculate_centroids(all_data, labels):
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), all_data.shape[1]))

    for i, label in enumerate(unique_labels):
        # Extract data points that belong to the current cluster
        cluster_points = all_data[labels == label]
        # Calculate the centroid as the mean of the cluster points
        centroids[i] = cluster_points.mean(axis=0)

    return centroids

class RBFNetwork(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        current_timestep,
        next_timestep,
        n_centers: int = 100,
        kappa: float = 1.0,
        clustering_method="kmeans",
        variance_aggregation="mean",
        lr=1e-2,
        now_reflow=0,
        datamodule=None,
    ):
        super().__init__()
        self.K = n_centers
        self.current_timestep = current_timestep
        self.next_timestep = next_timestep
        self.clustering_method = clustering_method
        if self.clustering_method == "kmeans":
            self.clustering_model = KMeans(n_clusters=self.K)
        elif self.clustering_method == "gmm":
            self.clustering_model = GaussianMixture(
                n_components=self.K, covariance_type="diag"
            )
        else:
            raise ValueError("Clustering method not supported")
        self.kappa = kappa
        self.variance_aggregation = variance_aggregation
        self.last_val_loss = 1
        self.lr = lr
        if self.clustering_method == "kmeans":
            self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        elif self.clustering_method == "gmm":
            self.W = torch.nn.Parameter(torch.rand(self.K, input_dim))

        self.now_reflow = now_reflow
        self.data_module = datamodule

    def on_before_zero_grad(self, *args, **kwargs):
        # clamp the weights to be non-negative
        self.W.data = torch.clamp(self.W.data, min=0.0001)

    def on_train_start(self):

        with torch.no_grad():
            if self.data_module is not None:
                if self.data_module.cluster_ambient:
                    all_data = torch.cat(
                        [
                            self.data_module.train_x0,
                            self.data_module.train_x1,
                        ]
                    )
                    x_decoded = torch.cat(
                        [
                            self.data_module.ambient_x0,
                            self.data_module.ambient_x1,
                        ]
                    )
                    print("all", all_data.shape)

                    self.all_data = all_data

                    self.all_data_shape = all_data.shape

                    print("Fitting Clustering model...")

                    x_decoded = transforms.Resize((64, 64))(x_decoded)
                    x_decoded = x_decoded.view(x_decoded.shape[0], -1)

                    self.clustering_model.fit(x_decoded)
                    print("Fitting Clustering model... Done")

                    if len(all_data.shape) > 2:
                        all_data = all_data.reshape(all_data.shape[0], -1)

                    clusters = calculate_centroids(
                        all_data, self.clustering_model.labels_
                    )

                    # encode clusters
                    self.C = torch.tensor(
                        clusters,
                        dtype=torch.float32,
                    ).to(self.device)

                else:
                    all_data = []
                    for batch in self.trainer.train_dataloader:
                        batch, _ = batch
                        all_data.append(batch)
                    all_data = torch.cat(all_data)
                    print("all", all_data.shape)
                    self.all_data = all_data
                    self.all_data_shape = all_data.shape

                    if len(all_data.shape) > 2:
                        all_data = all_data.reshape(all_data.shape[0], -1)

                    print("Fitting Clustering model...")
                    self.clustering_model.fit(all_data)
                    print("Fitting Clustering model... Done")

                    if self.clustering_method == "kmeans":
                        clusters = self.clustering_model.cluster_centers_
                    elif self.clustering_method == "gmm":
                        clusters = self.clustering_model.means_
                    self.C = torch.tensor(
                        clusters,
                        dtype=torch.float32,
                    ).to(self.device)
            else:
                all_data = []
                for batch in self.trainer.datamodule.train_dataloader():
                    metric_samples_batch_filtered = [
                        x
                        for i, x in enumerate(batch[0]["metric_samples"][0])
                        if i in [self.current_timestep, self.next_timestep]
                    ]
                    all_data = torch.cat(metric_samples_batch_filtered)

                    break


                print("Fitting Clustering model...")
                self.clustering_model.fit(all_data)
                print("Fitting Clustering model... Done")

                # clusters = self.kMeans_model.cluster_centers_
                if self.clustering_method == "kmeans":
                    clusters = self.clustering_model.cluster_centers_
                elif self.clustering_method == "gmm":
                    clusters = self.clustering_model.means_
                self.C = torch.tensor(
                    clusters,
                    dtype=torch.float32,
                ).to(self.device)
            # Calculate the bandwidths
            if self.clustering_method == "kmeans":
                labels = self.clustering_model.labels_
                sigmas = np.zeros((self.K, 1))
                for k in range(self.K):
                    inds_k = labels == k
                    points = all_data[inds_k, :]
                    c_k = clusters[k].reshape(-1, 1)
                    if self.variance_aggregation == "mean":
                        sigmas[k, :] = np.sqrt(
                            (
                                np.diag((points - c_k.T).T @ (points - c_k.T))
                                / points.shape[0]
                            ).mean()
                        )
                    elif self.variance_aggregation == "sum":
                        sigmas[k, :] = np.sqrt(
                            (
                                np.diag((points - c_k.T) @ (points - c_k.T).T)
                                / points.shape[0]
                            ).sum()
                        )
                    else:
                        raise ValueError("Variance aggregation method not supported")

            elif self.clustering_method == "gmm":
                sigmas = np.sqrt(self.clustering_model.covariances_)

            self.lamda = torch.tensor(
                0.5 / (self.kappa * sigmas) ** 2, dtype=torch.float32
            ).to(self.device)

            ###
            self.all_data = all_data

        if self.now_reflow == 0:
            torch.save(self.C, 'C.pt')
            torch.save(self.lamda, 'lamda.pt')


    def forward(self, x):

        if self.now_reflow != 0:
            self.C = torch.load('C.pt')
            self.lamda = torch.load('lamda.pt')

        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        dist2 = torch.cdist(x, self.C.to(x.device)) ** 2        # ----------------- #

        dist2 = dist2.to(self.device)
        self.lamda = self.lamda.to(self.device)

        self.phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        h_x = (self.W.to(x.device) * self.phi_x.to(x.device)).sum(dim=1)
        return h_x

    def training_step(self, batch, batch_idx):
        if self.data_module is not None:

            x, labels = batch
            h = self.forward(x)
            if labels is None:
                labels = 1
            loss = ((labels - h) ** 2).mean()
        else:
            main_batch = batch["train_samples"][0]
            metric_samples_batch_filtered = [
                x[0]
                for i, x in enumerate(main_batch)
                if i in [self.current_timestep, self.next_timestep]
            ]
            h = self.forward(torch.cat(metric_samples_batch_filtered))
            loss = ((1 - h) ** 2).mean()
        self.log(
            "MetricModel/train_loss_learn_metric",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch["val_samples"][0]
        metric_samples_batch_filtered = [
            x[0]
            for i, x in enumerate(main_batch)
            if i in [self.current_timestep, self.next_timestep]
        ]
        h = self.forward(torch.cat(metric_samples_batch_filtered))
        loss = ((1 - h) ** 2).mean()
        self.log(
            "MetricModel/val_loss_learn_metric",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.last_val_loss = loss.detach()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_metric(self, x, alpha=1, epsilon=1e-2, image_hx=False):
        # Calculate the RBF network's output for the given x
        if epsilon < 0:
            epsilon = (1 - self.last_val_loss.item()) / abs(epsilon)
        h_x = self.forward(x)
        if image_hx:  ## images
            h_x = 1 - torch.abs(1 - h_x)
            M_x = 1 / (h_x**alpha + epsilon)
        else:
            M_x = 1 / (h_x + epsilon) ** alpha
        return M_x
