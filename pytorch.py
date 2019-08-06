#%%
import torch

#%% [markdown]
# ### 3.1 Tensors Fundamentals
# - *torch.ones* and *torch.zeros* are used to create vector of 1 or 0
# - *torch.tensor* is used to create from a numeric list
# - Use [] to extract/assign element in a tensor by index
a = torch.ones(3)
aa = torch.zeros([3, 4])
b = torch.tensor([1.2, 2.3])
b[1] = 5.9
b[1]
c = torch.tensor([[1.0, 2.0, 3.0], [3.4, 5.6, 3.2]])

#%% [markdown]
# ### 3.3 Tensors and Storages
# - The storage underhood of tensor is a contiguous array
# - Stride is a tuple indicating the nubmer of elements in the storage taht have to be skipped when index is increased by 1 in each dimension
# - Accessing an element i, j: storage_offset + stride[0] * i + stride[1] * j
# - Changing sub-tensor will have side-effect on the original tensor, so we can clone the sub-tensor
# - Transpose of tensor have same storage but different stride, and not contiguous
# - We can obtain a new contiguous tensor from a non-contiguous tensor using contiguous method

#%% 
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()

second_point = points[1]
second_point.storage_offset()
second_point.size()
second_point.shape

points.stride()

second_point = points[1].clone()
second_point[0] = 10.0

points_t = points.t() # transpose the tensor
id(points.storage()) == id(points_t.storage()) # they have same storage
points_t.stride() # stride is transposed
points_t.is_contiguous()

points_t_cont = points_t.contiguous()

#%% [markdown]
# ### Numeric Types
# - torch.float32/torch.float, torch.float64/torch.double, torch.float16, torch.int8, torch.unint8, torch.int16, torch.int32, torch.int64
# - Specify numeric type using dtype argument
# - Specify numeric type using casting method

#%% 
double_points = torch.ones(10, 2, dtype = torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype = torch.short)

double_points = torch.ones(10, 2).to(torch.double)
short_points = torch.tensor([[1, 2], [3, 4]]).to(torch.short)

#%% [markdown]
# ### 3.5 Indexing tensors
# - some_list[:] all members in the list
# - some_list[1:4] from element 1 inclusive to element 4 exclusive
# - some_list[1:] from element 1 inclusive to the end of the list
# - some_list[:4] from the start of the list to element 4 exclusive
# - some_list[:-1] from the start of the list to one before the last element
# - some_list[1:4:2] from element 1 inclusive to element 4 exclusive in steps of 2

#%% [markdown]
# ### Numpy interoperability

#%% 
points = torch.ones(3, 4)
points_np = points.numpy()
points_np

points1 = torch.from_numpy(points_np)

#%% [markdown]
# ### The tensor API
# - creation ops: ones, zeros, from_numpy
# - Indexing, slicing, joining, mutating ops: transpose
# - Math ops
# - Random sampling: randn, normal
# - Serialization: load, save
# - Parallelism: set_num_threds

#%% [markdown]
# ### Exercise

#%% 
a = torch.tensor(range(9))
a.size()
a.storage_offset()
a.shape

# ??? why
b = a.view(3, 3)
b[1, 1]
c = b[1:, 1:]
c.size()
c.storage_offset()
c.stride()

torch.log(a.to(torch.float))
