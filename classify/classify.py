import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(pairwise_energy, similarity, num_classes=2):
    batch, height, width = similarity.shape
    
    assert batch == num_classes, str(batch) + ' ' + str(num_classes)
    # Assuming 'feature' is a softmax output of shape [Height, Width, num_classes]
    # 'text_feats' can be additional features of the same shape or could be incorporated differently depending on your needs
    
    # print('pairwise_energy', pairwise_energy.shape, pairwise_energy.min(), pairwise_energy.max())
    
    # Create a dense CRF model
    d = dcrf.DenseCRF2D(width, height, num_classes)
    
    # Convert the feature map to the unary potentials needed by the CRF
    
    # print('similarity', similarity.shape, similarity.min(), similarity.max(), similarity.dtype)
    
    unary = unary_from_softmax(similarity)
        
    # Set the unary potentials
    d.setUnaryEnergy(unary.reshape(num_classes, -1).copy(order='C'))

    # Optional: Add pairwise energy terms here, typically using the image itself or other modalities
    # This example adds a simple pairwise Gaussian term which can be modified or extended
    # d.addPairwiseGaussian(sxy=40, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=pairwise_energy, 
                           compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseEnergy(pairwise_energy)
    # You could integrate 'text_feats' as additional pairwise terms, e.g.,
    # d.addPairwiseBilateral(sxy=(10, 10), srgb=(5, 5, 5), rgbim=text_feats, compat=10)

    # Perform inference to get the refined segmentation
    Q = d.inference(5)  # number of iterations
    Q = np.array(Q)
    # print('Q', Q.shape, Q.min(), Q.max())
    refined_segmentation = Q.reshape((num_classes, height, width))

    return refined_segmentation[0]

# Example usage:
# feature should be a numpy array with shape [height, width, num_classes]
# text_feats should be shaped in a way that it can be integrated into the CRF process, depending on its nature
# refined_output = apply_crf(feature, text_feats, num_classes)



# def apply_depth_crf(depth, features):
#     batch, height, width = features.shape
    
#     assert batch == num_classes, str(batch) + ' ' + str(num_classes)
#     # Assuming 'feature' is a softmax output of shape [Height, Width, num_classes]
#     # 'text_feats' can be additional features of the same shape or could be incorporated differently depending on your needs
    
#     # print('pairwise_energy', pairwise_energy.shape, pairwise_energy.min(), pairwise_energy.max())
    
#     # Create a dense CRF model
#     d = dcrf.DenseCRF2D(width, height, num_classes)
    
#     # Convert the feature map to the unary potentials needed by the CRF
    
#     # print('similarity', similarity.shape, similarity.min(), similarity.max(), similarity.dtype)
    
#     unary = unary_from_softmax(features)
        
#     # Set the unary potentials
#     d.setUnaryEnergy(unary.reshape(num_classes, -1).copy(order='C'))

#     depth = depth.reshape(1, -1)
#     assert depth.shape[1] > depth.shape[0]
#     assert len(depth.shape) == 2     # -> (7, 100) = (feature dimensionality, npoints)
#     assert depth.dtype == np.float32     # -> dtype('float32')

#     dcrf.addPairwiseEnergy(depth, compat=10)

#     # Perform inference to get the refined segmentation
#     Q = d.inference(5)  # number of iterations
#     Q = np.array(Q)
#     # print('Q', Q.shape, Q.min(), Q.max())
#     refined_segmentation = Q.reshape((num_classes, height, width))

#     return refined_segmentation[0]