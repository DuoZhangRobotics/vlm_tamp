"""Bound the released proposer's clustering without changing DINO weights.

The upstream min/max normalization can divide by zero; its unbounded k-means
then loops forever on NaN. Constant dimensions carry zero information here.
Empty clusters are explicitly skipped, so a mask offers at most five proposals.
"""
import numpy as np


def normalized(values):
    import torch
    low=values.amin(0);span=values.amax(0)-low
    scaled=(values-low)/span.clamp_min(torch.finfo(values.dtype).eps)
    return torch.nan_to_num(scaled,nan=0.,posinf=0.,neginf=0.)


def bounded_kmeans(values,k,*,maximum_iterations=100,tolerance=1e-4):
    import torch
    values=torch.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)
    if values.ndim!=2 or len(values)<k or maximum_iterations<1:
        raise ValueError('Clustering needs enough finite samples and a positive iteration bound')
    first=int(torch.argmax(torch.linalg.vector_norm(values-values.mean(0),dim=1)))
    centres=[values[first]];minimum=torch.full((len(values),),float('inf'),device=values.device,dtype=values.dtype)
    for _ in range(k-1):
        minimum=torch.minimum(minimum,torch.sum((values-centres[-1])**2,dim=1))
        centres.append(values[int(torch.argmax(minimum))])
    centres=torch.stack(centres);converged=False
    for iteration in range(maximum_iterations):
        distances=torch.sum((values[:,None,:]-centres[None,:,:])**2,dim=2)
        labels=distances.argmin(1);updated=centres.clone()
        for label in range(k):
            members=values[labels==label]
            if len(members):updated[label]=members.mean(0)
        change=float(torch.linalg.vector_norm(updated-centres));centres=updated
        if change<tolerance:converged=True;break
    labels=torch.sum((values[:,None,:]-centres[None,:,:])**2,dim=2).argmin(1)
    return labels,centres,{'iterations':iteration+1,'maximum_iterations':maximum_iterations,
        'converged':converged,'occupied_clusters':int(len(torch.unique(labels))),
        'finite_centres':bool(torch.isfinite(centres).all())}


def cluster_features(self,points,features_flat,masks):
    import torch
    keypoints=[];pixels=[];groups=[];records=[]
    for group,mask in enumerate(masks):
        if np.mean(mask)>self.config['max_mask_ratio'] or mask.sum()<self.config['num_candidates_per_mask']:
            continue
        pixel=np.argwhere(mask);xyz=points[mask]
        features=features_flat[mask.reshape(-1)].double()
        if not torch.isfinite(features).all() or not np.isfinite(xyz).all():
            records.append({'group':group,'skipped':'nonfinite input'});continue
        _,_,v=torch.pca_lowrank(features,center=False)
        reduced=normalized(features@v[:,:3])
        spatial=normalized(torch.as_tensor(xyz,dtype=reduced.dtype,device=reduced.device))
        labels,centres,record=bounded_kmeans(torch.cat([reduced,spatial],dim=-1),self.config['num_candidates_per_mask'])
        record['group']=group;records.append(record)
        for label in range(len(centres)):
            indices=torch.nonzero(labels==label,as_tuple=False).flatten()
            if not len(indices):continue
            nearest=indices[torch.argmin(torch.linalg.vector_norm(reduced[indices]-centres[label,:3],dim=1))]
            index=int(nearest);keypoints.append(xyz[index]);pixels.append(pixel[index]);groups.append(group)
    self.bounded_cluster_records=records
    if not keypoints:raise ValueError('No finite keypoint clusters in predicted masks')
    return np.asarray(keypoints),np.asarray(pixels),np.asarray(groups)
