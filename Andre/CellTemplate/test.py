import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance

def visualizationOutGray(data, output, target, classes):
    #view first image in each batch w/ its prediction and label
    ig = plt.figure()
    output_cpu = output.to(torch.device("cpu"))
    target_cpu = target.to(torch.device("cpu"))
    data_cpu = data.to(torch.device("cpu"))
    output_idx = (np.argmax(output_cpu[0], axis=0)) #reverse one hot
    cls = classes[output_idx]
    plt.title("Prediction = " + str(cls) + " | Actual = " + str(classes[target_cpu[0].numpy()]) )
    img = data_cpu[0]
    plt.imshow(np.transpose(np.reshape(img, (1,28,28)), (1,2,0)).squeeze(), cmap = 'gray') # realign 
    
def visualizationOutColor(data, output, target, classes):
    #view first image in each batch w/ its prediction and label
    fig = plt.figure()
    output_cpu = output.to(torch.device("cpu"))
    target_cpu = target.to(torch.device("cpu"))
    data_cpu = data.to(torch.device("cpu"))
    idx = (np.argmax(output_cpu[0], axis=0))
    cls = classes[idx]
    plt.title("Prediction = " + str(cls) + " | Actual = " + str(classes[target_cpu[0].numpy()]) )
    img = data_cpu[0]
    plt.imshow(np.transpose(np.reshape(img,(3, 32,32)), (1,2,0))) #un-normalize and realign    
                
def main(config, resume):
    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader_test', config)
    '''
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    '''

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU to test")
        
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
