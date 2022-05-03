import numpy as np
import mxnet as mx
import os
import solvers.zo.exp_grad as ExpGrad
import solvers.zo.pgd as PGD
import solvers.zo.ada_exp_grad as AdaExpGrad

def l1_norm(x):
    return np.linalg.norm(x.flatten(),ord=1)

def trace_norm(x):
    s=np.linalg.svd(x,full_matrices=False,compute_uv=False)
    return np.linalg.norm(s.flatten(),ord=1)
    
def main(args):
    num_samples=args['num_samples']
    num_iter=args['num_iterations']
    device=args['device']
    l1=args['l1']
    l2=args['l2']
    mode=args['mode']
    data=args['data']
    kappa=args['kappa']
    seed=args['seed']
    result_path=args['path']
    root=args['imagenet']
    alg_name=args['alg']
    if alg_name=='PGD':
        algs=[PGD]
        lr=[1.0,0.1,0.01,0.001,0.0001]
    elif alg_name=='ExpMD':
        algs=[ExpGrad]
        lr=[1.0,0.1,0.01,0.001,0.0001]
    else:
        algs=[AdaExpGrad]
        lr=[2.0]
    get_l1=l1_norm
    if device <0:
        ctx=mx.cpu()
    else:
        ctx=mx.gpu(device)
    
    if data=="CIFAR":
        import loss.cifer_resnet as model
        max_index=10000
        num_classes=10
    elif data=="MNIST":
        import loss.mnist_lenet as model
        max_index=10000
        num_classes=10
    elif data=="IMAGENET":
        import loss.imagenet_resnet as model
        max_index=50000
        num_classes=1
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    for eta in lr:
        for alg in algs:
            file_path=data+'_'+mode+'_'+alg.__name__+'_'+str(eta)
            file_path=os.path.join(result_path,file_path)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            print("start algorithm ",alg.__name__)
            train_loss=[]
            attack_loss=[]
            average_train_loss=np.zeros(num_iter+1)
            attack_array=np.zeros(num_iter+1)
            for class_idx in range(0,num_classes):
                train_loss_array=np.zeros(num_iter+1)
                attack_array=np.zeros(num_iter+1)
                print("class ",class_idx)
                np.random.seed(seed+class_idx)   
                shuffled_indices=np.random.permutation(max_index)
                current_index=0
                for sample_index in range(0,num_samples):
                    print("sample",sample_index)
                    cem=None
                    found=False
                    while not found and current_index<max_index:            
                        if data in ["CIFER","MNIST"]:
                            cem=model.CEM(index=shuffled_indices[current_index],ctx=ctx,kappa=kappa)
                        elif data=="IMAGENET":
                            cem=model.CEM(index=shuffled_indices[current_index],ctx=ctx,kappa=kappa,folder_path=root)
                        current_index+=1
                        if data== data in ["CIFER","MNIST"] and (cem.label== class_idx and cem.correct):
                            found=True
                            break
                        elif data=="IMAGENET" and cem.correct:                        
                            found=True
                            break
                    if not found:
                        print("not enough sample of class: ", class_idx)
                        return
                    if mode=="PP":
                        func=model.PP_Loss(cem)
                        init=cem.pp_init
                        upper=cem.pp_upper
                        lower=cem.pp_lower
                    else:
                        func=model.PN_Loss(cem)
                        init=cem.pn_init
                        upper=cem.pn_upper
                        lower=cem.pn_lower
                    regl1=get_l1(init)
                    regl2=np.linalg.norm(init.flatten(),ord=2)**2
                    attack=func(init)
                    print('iteration: ',0,' attack: ', attack,' loss: ',1.0,' l1: ',regl1,' l2: ',regl2 )
                    init_loss=attack+l1*regl1+0.5*l2*regl2+kappa
                    train_loss.append(1.0)
                    attack_loss.append(attack)
                    def callback(res):
                        regl1=get_l1(res.x)
                        regl2=np.linalg.norm(res.x.flatten(),ord=2)**2
                        cur_loss=res.func+l1*regl1+0.5*l2*regl2+kappa
                        if res.nit%(num_iter//10)==0:
                            print('iteration: ',res.nit,' attack: ', res.func,' loss: ',cur_loss/init_loss,' l1: ',regl1,' l2: ',regl2 )
                        train_loss.append(cur_loss/init_loss)
                        attack_loss.append(res.func)
                    alg.fmin(func=func, x0=init, lower=lower, upper=upper,l1=l1,l2=l2,maxfev=num_iter,callback=callback,epoch_size=1,eta=eta)
                    train_loss_array+=((1.0/num_samples)*np.array(train_loss))
                    attack_array+=((1.0/num_samples)*np.array(attack_loss))
                    loss_file = os.path.join(file_path,'class_'+str(class_idx)+str(sample_index)+'_loss.csv')
                    attack_file = os.path.join(file_path,'class_'+str(class_idx)+str(sample_index)+'_attack.csv')
                    np.savetxt(loss_file, np.array(train_loss), delimiter=",")
                    np.savetxt(attack_file, np.array(attack_loss), delimiter=",")
                    train_loss=[]
                    attack_loss=[]
            average_train_loss+=(train_loss_array/10.0)
            loss_file = os.path.join(file_path,'average_loss.csv')
            np.savetxt(loss_file, average_train_loss, delimiter=",")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=20)
    parser.add_argument("-t", "--num_iterations", type=int, default=200)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--l1", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--mode", choices=["PN", "PP"], default="PN")
    parser.add_argument("--data", choices=[ "CIFAR","MNIST","IMAGENET"], default="MNIST")
    parser.add_argument("--alg", choices=[ "PGD","ExpMD","AdaExpMD"], default="AdaExpMD")
    parser.add_argument("--kappa", type=float, default=2)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--path", type=str, default='experiment_results')
    parser.add_argument("--imagenet", type=str, default='./datasets/imagenet')
    args = vars(parser.parse_args())
    main(args)
