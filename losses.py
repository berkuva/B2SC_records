import torch
import torch.nn.functional as F
import pdb
import torch.nn as nn

def gmm_loss(gmm_weights_param, true_gmm_fractions=None, device='cuda'):
    # Default values for the true GMM fractions if not provided
    if true_gmm_fractions is None:
        true_gmm_fractions = torch.tensor([0.187, 0.129, 0.113, 0.113, 0.146, 0.193, 0.10, 0.013, 0.006])

    # Move true_gmm_fractions to the specified device
    true_gmm_fractions = true_gmm_fractions.to(device)
    
    # Apply softmax to gmm_weights_param
    # gmm_weights = torch.nn.Softmax(dim=0)(gmm_weights_param)
    
    # Calculate the MSE loss
    loss = nn.MSELoss()(gmm_weights_param, true_gmm_fractions)
    
    return loss


def KLDiv(mu1, logvar1,
                  mu2, logvar2,
                  mu3, logvar3,
                  mu4, logvar4,
                  mu5, logvar5,
                  mu6, logvar6,
                  mu7, logvar7,
                  mu8, logvar8,
                  mu9, logvar9,
                  gmm_weights_param):
    weight1 = gmm_weights_param[0]
    weight2 = gmm_weights_param[1]
    weight3 = gmm_weights_param[2]
    weight4 = gmm_weights_param[3]
    weight5 = gmm_weights_param[4]
    weight6 = gmm_weights_param[5]
    weight7 = gmm_weights_param[6]
    weight8 = gmm_weights_param[7]
    weight9 = gmm_weights_param[8]


    KLD1 = weight1 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    KLD2 = weight2 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    KLD3 = weight3 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp())
    KLD4 = weight4 * torch.sum(1 + logvar4 - mu4.pow(2) - logvar4.exp())
    KLD5 = weight5 * torch.sum(1 + logvar5 - mu5.pow(2) - logvar5.exp())
    KLD6 = weight6 * torch.sum(1 + logvar6 - mu6.pow(2) - logvar6.exp())
    KLD7 = weight7 * torch.sum(1 + logvar7 - mu7.pow(2) - logvar7.exp())
    KLD8 = weight8 * torch.sum(1 + logvar8 - mu8.pow(2) - logvar8.exp())
    KLD9 = weight9 * torch.sum(1 + logvar9 - mu9.pow(2) - logvar9.exp())
    # pdb.set_trace()
    KLD = KLD1 + KLD2 + KLD3 + KLD4 + KLD5 + KLD6 + KLD7 + KLD8 + KLD9

    # KLD = torch.clamp(KLD, min=-100, max=100)
    
    return KLD




def bulk_loss(scmu1, sclogvar1,
              scmu2, sclogvar2,
              scmu3, sclogvar3,
              scmu4, sclogvar4,
              scmu5, sclogvar5,
              scmu6, sclogvar6,
              scmu7, sclogvar7,
              scmu8, sclogvar8,
              scmu9, sclogvar9,
              bulkmu1, bulklogvar1,
              bulkmu2, bulklogvar2,
              bulkmu3, bulklogvar3,
              bulkmu4, bulklogvar4,
              bulkmu5, bulklogvar5,
              bulkmu6, bulklogvar6,
              bulkmu7, bulklogvar7,
              bulkmu8, bulklogvar8,
              bulkmu9, bulklogvar9):
    loss1 = nn.MSELoss()(scmu1, bulkmu1) + nn.MSELoss()(sclogvar1, bulklogvar1)
    loss2 = nn.MSELoss()(scmu2, bulkmu2) + nn.MSELoss()(sclogvar2, bulklogvar2)
    loss3 = nn.MSELoss()(scmu3, bulkmu3) + nn.MSELoss()(sclogvar3, bulklogvar3)
    loss4 = nn.MSELoss()(scmu4, bulkmu4) + nn.MSELoss()(sclogvar4, bulklogvar4)
    loss5 = nn.MSELoss()(scmu5, bulkmu5) + nn.MSELoss()(sclogvar5, bulklogvar5)
    loss6 = nn.MSELoss()(scmu6, bulkmu6) + nn.MSELoss()(sclogvar6, bulklogvar6)
    loss7 = nn.MSELoss()(scmu7, bulkmu7) + nn.MSELoss()(sclogvar7, bulklogvar7)
    loss8 = nn.MSELoss()(scmu8, bulkmu8) + nn.MSELoss()(sclogvar8, bulklogvar8)
    loss9 = nn.MSELoss()(scmu9, bulkmu9) + nn.MSELoss()(sclogvar9, bulklogvar9)

    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9

    return loss