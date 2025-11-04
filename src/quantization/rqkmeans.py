import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
import time

NUM_LOGGER_STEPS = 100


def get_time(device, sync=False):
    if (device is not None) and sync:
        torch.cuda.synchronize(device)
    return time.perf_counter()


class KMeans:
    def __init__(
        self,
        n_clusters,
        batch_size,
        window_size=32,
        max_iter=300,
        tol=1e-4,
        verbose=1,
        random_state=None,
        gpu_preload=True
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.cluster_centers_ = None
        self.gpu_preload = gpu_preload
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f'cuda')
        self.logger_step = max(self.max_iter // NUM_LOGGER_STEPS, 1)
        self.kmpp_logger_step = max(self.n_clusters // NUM_LOGGER_STEPS, 1)
        self._eps = 1e-12

    def _compute_distances(self, batch, centroids):
        b2 = (batch.float() * batch.float()).sum(dim=1, keepdim=True)
        c2 = (centroids.float() * centroids.float()).sum(dim=1).unsqueeze(0)
        if batch.is_cuda:
            dot = (batch.half() @ centroids.t().half()).float()
        else:
            dot = (batch.float() @ centroids.t().float())
        return torch.relu(b2 + c2 - 2.0 * dot)

    def _init_centroids(self, X):
        assert X.shape[0] >= self.n_clusters, "Not implemented"
        X = X[torch.randperm(X.size()[0])]
        C = X[:self.n_clusters].to(self.device)
        return C

    def _global_sum_int(self, value: int) -> int:
        value = torch.tensor([value], dtype=torch.long, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return int(value.item())

    def _select_global_topm_mask_inplace(self, gather_d_flat, sel_mask_flat, m, owner):
        if self.rank == owner:
            gd = gather_d_flat.view(self.world_size, m).to(torch.float32)
            gd_masked = torch.where(torch.isfinite(gd), gd, torch.full_like(gd, float('inf')))
            ords = torch.topk(gd_masked.view(-1), k=m, largest=False)[1]
            sel_mask_flat.zero_()
            sel_mask_flat[ords] = 1
        dist.broadcast(sel_mask_flat, src=owner)

    def _fix_dead_centroids(self, X, min_dists, centroids, cluster_counts):
        mask = (cluster_counts == 0).int()
        dist.all_reduce(mask, op=dist.ReduceOp.SUM)
        dead_indices = (mask > 0).nonzero(as_tuple=True)[0]
        if dead_indices.numel() == 0:
            return centroids

        used_local = torch.zeros(X.shape[0], dtype=torch.bool, device=self.device)
        for dead_j in dead_indices.tolist():
            if (~used_local).any():
                min_dists[used_local] = -float("inf")
                local_max_val, local_max_idx = min_dists.max(0)
                local_max_val = local_max_val.item()
                local_max_idx = local_max_idx.item()
                local_max_feat = X[local_max_idx].to(self.device)
            else:
                local_max_val = -float("inf")
                local_max_feat = torch.zeros_like(centroids[0])
                local_max_idx = -1

            vals = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
            feats = [torch.zeros_like(local_max_feat).unsqueeze(0) for _ in range(self.world_size)]
            idxs = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(vals, torch.tensor([local_max_val], device=self.device))
            dist.all_gather(feats, local_max_feat.unsqueeze(0))
            dist.all_gather(idxs, torch.tensor([local_max_idx], dtype=torch.long, device=self.device))

            vals = torch.cat(vals)
            idxs = torch.cat(idxs)
            best_rank = vals.argmax().item()
            winner_feat = feats[best_rank]
            win_local_idx = idxs[best_rank]
            if (self.rank == best_rank) and (win_local_idx != -1):
                used_local[win_local_idx] = True
            centroids[dead_j] = winner_feat

        return centroids

    def _core_step(self, X, centroids):
        cluster_sums = torch.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype, device=self.device)
        cluster_counts = torch.zeros(self.n_clusters, dtype=torch.float32, device=self.device)
        all_min_dists = torch.full((X.shape[0],), float('inf'), dtype=X.dtype, device=self.device)
        for idx in range(0, X.shape[0], self.batch_size):
            batch = X[idx:idx + self.batch_size].to(self.device)
            dmat = self._compute_distances(batch, centroids)
            labels = torch.argmin(dmat, dim=1)
            cluster_sums.index_add_(0, labels, batch)
            counts = torch.bincount(labels, minlength=self.n_clusters).to(torch.float32)
            cluster_counts += counts
            all_min_dists[idx:idx + self.batch_size] = torch.min(dmat, dim=1)[0]

        return cluster_sums, cluster_counts, all_min_dists

    def fit(self, X):
        np.random.seed(self.random_state + self.rank)
        torch.manual_seed(self.random_state + self.rank)
        torch.cuda.manual_seed(self.random_state + self.rank)
        torch.cuda.manual_seed_all(self.random_state + self.rank)

        X = X.to(self.device) if self.gpu_preload else X
        centroids = self._init_centroids(X)
        dist.broadcast(centroids, src=0)

        for i in range(self.max_iter):
            t0 = get_time(self.device, sync=False)
            cluster_sums_local, cluster_counts_local, all_min_dists_local = self._core_step(X, centroids)
            pack = torch.empty((self.n_clusters, X.shape[1] + 1), dtype=torch.float32, device=self.device)
            pack[:, :X.shape[1]] = cluster_sums_local
            pack[:, X.shape[1]] = cluster_counts_local
            dist.all_reduce(pack, op=dist.ReduceOp.SUM)
            cluster_sums = pack[:, :X.shape[1]]
            cluster_counts = pack[:, X.shape[1]]
            self.cluster_centers_ = centroids.clone()
            mask = (cluster_counts > 0)
            if mask.any():
                self.cluster_centers_[mask] = cluster_sums[mask] / cluster_counts[mask].unsqueeze(1)
            self.cluster_centers_ = self._fix_dead_centroids(X, all_min_dists_local, self.cluster_centers_, cluster_counts)
            shift = torch.norm(centroids - self.cluster_centers_, dim=1).max().item()
            centroids = self.cluster_centers_

            if self.verbose and (self.rank == 0) and (i % self.logger_step == 0):
                t1 = get_time(self.device, sync=True)
                logger.info(f"Epoch {i+1}/{self.max_iter} done in {t1 - t0:.2f} sec | Max centroid shift: {shift:.6f}")
            if shift < self.tol:
                break

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted yet!")

        labels = torch.empty(X.shape[0], dtype=torch.int32)
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, self.batch_size + start)
            batch = X[start:end]
            dists = self._compute_distances(batch.to(self.device), self.cluster_centers_)
            labels[start:end] = torch.argmin(dists, dim=1).cpu()
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


class RQKMeans:
    def __init__(self, num_clusters, num_ids, batch_size, window_size=32, max_iter=300, tol=1e-4, verbose=0, random_state=42, gpu_preload=True):
        self.models = [
            KMeans(
                n_clusters=num_clusters,
                batch_size=batch_size,
                window_size=window_size,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                random_state=random_state + i,
                gpu_preload=gpu_preload
            ) for i in range(num_ids)
        ]

    def fit(self, X, y=None):
        residual = X.clone()
        for model in self.models:
            y = model.fit_predict(residual)
            residual.sub_(model.cluster_centers_.cpu().index_select(0, y))
        return self

    def predict(self, X):
        residual = X.clone()
        labels = torch.empty((X.shape[0], len(self.models)), dtype=torch.long)
        for model_index, model in enumerate(self.models):
            y = model.predict(residual)
            labels[:, model_index] = y
            C = model.cluster_centers_.cpu().index_select(0, y)
            residual.sub_(C)
        return labels

    def iter_sum_centroids(self, y):
        centers_per_model = [model.cluster_centers_.cpu() for model in self.models]
        for i in range(y.shape[0]):
            labels_row = y[i]
            sum_vector = centers_per_model[0][int(labels_row[0].item())].clone()
            for model_index in range(1, len(self.models)):
                sum_vector.add_(centers_per_model[model_index][int(labels_row[model_index].item())])
            yield sum_vector
