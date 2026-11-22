import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k, common_choice=False):
    if common_choice:
        idx = random.choices(range(len(obj_list)), k=k) 
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)

# **************** Class-aware samplers for Scoliosis1K (exp_2) ****************
CLASS_ORDER = ['negative', 'neutral', 'positive']


def _broadcast_list(values):
    """Rank 0 decides, every rank gets the same list (same idea as sync_random_sample_list)."""
    idx = torch.tensor(values, dtype=torch.long)
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    return idx.tolist()


class ClassAwareSampler(tordata.sampler.Sampler):
    """Triplet-style P x K batches whose class mix is controlled.

    mode='stratified': every batch holds a fixed number of children per class.
        class_counts=[n_neg, n_neu, n_pos] (sum = P), or None for an even split of P;
        the remainder of P // 3 goes to random classes, so the mix is balanced on average.
    mode='upsample': children are drawn at random (no repeat inside a batch) with weight
        n_class ** -power. power=0 is the natural sampler, power=1 is full upsampling
        (every class equally likely), power=0.5 is the square-root compromise.
    K copies of each child are taken with replacement (Scoliosis1K has 1 sequence per child),
    exactly like TripletSampler.
    class_prior: the class distribution the loss sees on average; the LA loss uses it.
    """

    def __init__(self, dataset, batch_size, mode='stratified', class_counts=None, power=1.0,
                 batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(batch_size) != 2:
            raise ValueError("batch_size should be (P x K) not {}".format(batch_size))
        if mode not in ('stratified', 'upsample'):
            raise ValueError("mode should be 'stratified' or 'upsample', not {}".format(mode))
        self.mode, self.power, self.batch_shuffle = mode, float(power), batch_shuffle
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        if (batch_size[0] * batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))

        self.pids_by_class = {c: [] for c in CLASS_ORDER}
        for pid in dataset.label_set:
            cls = dataset.types_list[dataset.indices_dict[pid][0]].lower()
            self.pids_by_class[cls].append(pid)
        self.n_class = [len(self.pids_by_class[c]) for c in CLASS_ORDER]
        if min(self.n_class) == 0:
            raise ValueError("every class needs at least one child, got {}".format(self.n_class))
        self.class_counts = class_counts
        if mode == 'stratified' and class_counts is not None and sum(class_counts) != batch_size[0]:
            raise ValueError("sum(class_counts) must equal P = {}".format(batch_size[0]))

        self.all_pids = [p for c in CLASS_ORDER for p in self.pids_by_class[c]]
        w = [n ** -self.power for n in self.n_class for _ in range(n)]
        self.child_weights = torch.tensor(w, dtype=torch.double)

    @property
    def class_prior(self):
        if self.mode == 'stratified':
            if self.class_counts is None:
                return [1 / 3] * 3
            return [c / sum(self.class_counts) for c in self.class_counts]
        q = [n ** (1 - self.power) for n in self.n_class]
        return [v / sum(q) for v in q]

    def _counts(self):
        if self.class_counts is not None:
            return list(self.class_counts)
        P = self.batch_size[0]
        counts = [P // 3] * 3
        for c in random.sample(range(3), P % 3):
            counts[c] += 1
        return counts

    def _pick_children(self):
        if self.mode == 'stratified':
            pids = []
            for c, n in zip(CLASS_ORDER, self._counts()):
                pool = self.pids_by_class[c]
                pids += random.sample(pool, n) if n <= len(pool) else random.choices(pool, k=n)
            return [self.all_pids.index(p) for p in pids]
        return torch.multinomial(self.child_weights, self.batch_size[0], replacement=False).tolist()

    def __iter__(self):
        while True:
            child_idx = _broadcast_list(self._pick_children())
            sample_indices = []
            for i in child_idx:
                indices = self.dataset.indices_dict[self.all_pids[i]]
                sample_indices += sync_random_sample_list(indices, k=self.batch_size[1])
            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(sample_indices, len(sample_indices))
            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size / self.world_size)) * self.world_size
            sample_indices += sample_indices[:(total_batch_size - len(sample_indices))]
            yield sample_indices[self.rank:total_size:self.world_size]

    def __len__(self):
        return len(self.dataset)
