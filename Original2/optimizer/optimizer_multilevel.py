import torch.optim as optim

def set_optimizer(opt, model):
    # print("model.parameters(): ", model.parameters())
    # for param in model.parameters():
    #     print("param: ", param)
    # print("model.state_dict(): ", model.state_dict())
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    # optimizer = optim.SGD(model.pseudo_targets.base_fc.parameters(),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay)
    return optimizer


