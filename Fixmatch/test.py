import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
from dataset.data_getter import get_cifar10
from utils import AverageMeter, accuracy
from tqdm import tqdm
import models.wideresnet as models
import time
import logging

import gc
gc.collect()
torch.cuda.empty_cache()
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

def main():
    # gpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # load model
    model = models.build_wideresnet(depth=28,
                                widen_factor=2,
                                dropout=0,
                                num_classes=10)
    model.to(device)
    checkpoint_path = "results/cifar10@4000.5/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded!")

    # load test data
    labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10()
    test_loader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=8,
    num_workers=1)

    # test
    test(test_loader, model, device)

# test function
def test(test_loader, model, device):
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            # test_loader = tqdm(test_loader, disable=True)

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
        #     test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
        #         batch=batch_idx + 1,
        #         iter=len(test_loader),
        #         data=data_time.avg,
        #         bt=batch_time.avg,
        #         loss=losses.avg,
        #         top1=top1.avg,
        #         top5=top5.avg,
        #     ))

        # test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()