import tensorflow as tf
import numpy as np
import os
import pickle
from model import create_f_model, create_z_model, create_v_model
from utils import *

def train_adversarial_model_f(ds, cfg, exp_name, run, print_results=False):
    
    tf.keras.utils.set_random_seed(run)
    
    trn, val, tst = ds.get_dataset_in_tensor(batch_size=cfg['batch_size'])
    if cfg['include_u']:
        x_dim = trn.element_spec[0].shape[1]+trn.element_spec[1].shape[1]
    else:
        x_dim = trn.element_spec[0].shape[1]
    u_dim = trn.element_spec[1].shape[1]
    y_dim = trn.element_spec[2].shape[1]
    
    fl = cfg['f_hidden_layers']
    fu = cfg['f_hidden_units']
    vl = cfg['v_hidden_layers']
    vu = cfg['v_hidden_units']
    z_layer = cfg['z_layer_f']
    reg = cfg['reg']
    
    f_model = create_f_model(fl, fu, x_dim, y_dim)
    z_model = create_z_model(f_model, z_layer)
    z_dim = f_model.layers[z_layer].output.shape[1]
    v0_model = create_v_model(vl, vu, z_dim, u_dim)
    v1_model = create_v_model(vl, vu, z_dim, u_dim)
    
    lr_schedule_f = tf.keras.optimizers.schedules.ExponentialDecay(cfg['initial_lr_f'],
                                                                   decay_steps=1000,
                                                                   decay_rate=0.98,
                                                                   staircase=True)
    f_optimizer = tf.keras.optimizers.Adam(cfg['initial_lr_f'], beta_1=0.5)
    v0_optimizer = tf.keras.optimizers.SGD(cfg['initial_lr_v'])
    v1_optimizer = tf.keras.optimizers.SGD(cfg['initial_lr_v'])
    
    f_trn_acc_metric = tf.keras.metrics.BinaryAccuracy()
    f_val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    if u_dim == 1:
        v0_trn_acc_metric = tf.keras.metrics.BinaryAccuracy()
        v0_val_acc_metric = tf.keras.metrics.BinaryAccuracy()
        v1_trn_acc_metric = tf.keras.metrics.BinaryAccuracy()
        v1_val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    else:
        v0_trn_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        v0_val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        v1_trn_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        v1_val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    for epoch in range(cfg['epochs_fv']):
        f_trn_loss = tf.Variable(0.0)
        f_val_loss = tf.Variable(0.0)
        v0_trn_loss = tf.Variable(0.0)
        v0_val_loss = tf.Variable(0.0)
        v1_trn_loss = tf.Variable(0.0)
        v1_val_loss = tf.Variable(0.0)
        
        @tf.function
        def train_step_fv_model(x, u, y):
            with tf.GradientTape() as f_tape:
                preds_y = f_model(x)
                z_pred = z_model(x)
                z_pred_0 = tf.boolean_mask(z_pred, mask_0)
                z_pred_1 = tf.boolean_mask(z_pred, mask_1) 
                preds_u0 = v0_model(z_pred_0)
                preds_u1 = v1_model(z_pred_1)
                u0 = tf.boolean_mask(u, mask_0)
                u1 = tf.boolean_mask(u, mask_1)

                if u_dim == 1:
                    f_loss = BCE_loss(y, preds_y) - reg * (BCE_loss(u0, preds_u0) + BCE_loss(u1, preds_u1))
                else:
                    f_loss = BCE_loss(y, preds_y) - reg * (CCE_loss(u0, preds_u0) + CCE_loss(u1, preds_u1))
                f_trn_loss.assign_add(f_loss)

            gradients_of_f = f_tape.gradient(f_loss, f_model.trainable_variables)
            f_optimizer.apply_gradients(zip(gradients_of_f, f_model.trainable_variables))
            f_trn_acc_metric.update_state(y, preds_y)

            with tf.GradientTape() as v0_tape:
                z_pred = z_model(x)
                z_pred_0 = tf.boolean_mask(z_pred, mask_0)
                preds_u0 = v0_model(z_pred_0)
                u0 = tf.boolean_mask(u, mask_0)

                if u_dim == 1:
                    v0_loss = BCE_loss(u0, preds_u0)
                else:
                    v0_loss = CCE_loss(u0, preds_u0)
                v0_trn_loss.assign_add(v0_loss)

            gradients_of_v0 = v0_tape.gradient(v0_loss, v0_model.trainable_variables)
            v0_optimizer.apply_gradients(zip(gradients_of_v0, v0_model.trainable_variables))
            v0_trn_acc_metric.update_state(u0, preds_u0)  

            with tf.GradientTape() as v1_tape:
                z_pred = z_model(x)
                z_pred_1 = tf.boolean_mask(z_pred, mask_1)
                preds_u1 = v1_model(z_pred_1)
                u1 = tf.boolean_mask(u, mask_1)

                if u_dim == 1:
                    v1_loss = BCE_loss(u1, preds_u1)
                else:
                    v1_loss = CCE_loss(u1, preds_u1)
                v1_trn_loss.assign_add(v1_loss)

            gradients_of_v1 = v1_tape.gradient(v1_loss, v1_model.trainable_variables)
            v1_optimizer.apply_gradients(zip(gradients_of_v1, v1_model.trainable_variables))
            v1_trn_acc_metric.update_state(u1, preds_u1)  

        @tf.function
        def valid_step_fv_model(x, u, y):
            preds_y = f_model(x)
            z_pred = z_model(x)
            z_pred_0 = tf.boolean_mask(z_pred, mask_0)
            z_pred_1 = tf.boolean_mask(z_pred, mask_1)
            preds_u0 = v0_model(z_pred_0)
            preds_u1 = v1_model(z_pred_1)
            u0 = tf.boolean_mask(u, mask_0)
            u1 = tf.boolean_mask(u, mask_1)

            if u_dim == 1:
                f_loss = BCE_loss(y, preds_y) - reg * (BCE_loss(u0, preds_u0) + BCE_loss(u1, preds_u1))
            else:
                f_loss = BCE_loss(y, preds_y) - reg * (CCE_loss(u0, preds_u0) + CCE_loss(u1, preds_u1))
            f_val_loss.assign_add(f_loss)
            y_pred = tf.cast(preds_y > 0.5, dtype=tf.float32)
            f_val_acc_metric.update_state(y, y_pred)

            if u_dim == 1:
                v0_loss =  BCE_loss(u0, preds_u0)
                v1_loss =  BCE_loss(u1, preds_u1)
            else:
                v0_loss =  CCE_loss(u0, preds_u0)
                v1_loss =  CCE_loss(u1, preds_u1)
            v0_val_loss.assign_add(v0_loss)  
            v1_val_loss.assign_add(v1_loss)  
            v0_val_acc_metric.update_state(u0, preds_u0)
            v1_val_acc_metric.update_state(u1, preds_u1)
        
        for (x, u, y) in trn:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)

            mask_0 = tf.squeeze(tf.equal(y, 0), axis=1)
            mask_1 = tf.squeeze(tf.equal(y, 1), axis=1)

            train_step_fv_model(x, u, y)
            
        f_trn_acc = f_trn_acc_metric.result()
        f_trn_acc_metric.reset_state()
        v0_trn_acc = v0_trn_acc_metric.result()
        v0_trn_acc_metric.reset_state()
        v1_trn_acc = v1_trn_acc_metric.result()
        v1_trn_acc_metric.reset_state()
            
        for (x, u, y) in val:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)

            mask_0 = tf.squeeze(tf.equal(y, 0), axis=1)
            mask_1 = tf.squeeze(tf.equal(y, 1), axis=1)

            valid_step_fv_model(x, u, y)  
            
        f_val_acc = f_val_acc_metric.result()
        f_val_acc_metric.reset_state()
        v0_val_acc = v0_val_acc_metric.result()
        v0_val_acc_metric.reset_state()
        v1_val_acc = v1_val_acc_metric.result()
        v1_val_acc_metric.reset_state()

        
        if print_results:
            print(f"Epoch: {epoch}")
            print(f"f train loss: {f_trn_loss.numpy()/len(trn):.3f}, f valid loss: {f_val_loss.numpy()/len(val):.3f}, f train acc: {f_trn_acc.numpy():.3f}, f valid acc: {f_val_acc.numpy():.3f}")
            print(f"v0 train loss: {v0_trn_loss.numpy()/len(trn):.3f}, v0 valid loss: {v0_val_loss.numpy()/len(val):.3f}, v0 train acc: {v0_trn_acc.numpy():.3f}, v0 valid acc: {v0_val_acc.numpy():.3f}")
            print(f"v1 train loss: {v1_trn_loss.numpy()/len(trn):.3f}, v1 valid loss: {v1_val_loss.numpy()/len(val):.3f}, v1 train acc: {v1_trn_acc.numpy():.3f}, v1 valid acc: {v1_val_acc.numpy():.3f}")
            
        if not os.path.exists(f'{exp_name}/saved_models/run_{run+1}'):
            os.makedirs(f'{exp_name}/saved_models/run_{run+1}')
    
    f_model.save(f'{exp_name}/saved_models/run_{run+1}/v_fair_fl_{fl}_fu_{fu}_vl_{vl}_vu_{vu}_z_{z_layer}_reg_{reg}_model.keras')
    v0_model.save(f'{exp_name}/saved_models/run_{run+1}/v_fair_v0_vl_{vl}_vu_{vu}_fl_{fl}_fu_{fu}_z_{z_layer}_reg_{reg}_model.keras')
    v1_model.save(f'{exp_name}/saved_models/run_{run+1}/v_fair_v1_vl_{vl}_vu_{vu}_fl_{fl}_fu_{fu}_z_{z_layer}_reg_{reg}_model.keras')
            
def train_adversarial_model_adv(ds, cfg, exp_name, run, print_results=False):
    
    tf.keras.utils.set_random_seed(run)

    trn, val, tst = ds.get_dataset_in_tensor(batch_size=cfg['batch_size'])
    u_dim = trn.element_spec[1].shape[1]
    
    @tf.function
    def train_step_adv_model(x, u):
        with tf.GradientTape() as adv_tape:
            z_pred = z_model(x)
            preds = adv_model(z_pred)
            if u_dim == 1:
                adv_loss = BCE_loss(u, preds)
            else:
                adv_loss = CCE_loss(u, preds)

        gradients_of_adv = adv_tape.gradient(adv_loss, adv_model.trainable_variables)
        adv_optimizer.apply_gradients(zip(gradients_of_adv, adv_model.trainable_variables))
        trn_loss_metric.update_state(u, preds)
        trn_acc_metric.update_state(u, preds)
        

    @tf.function
    def valid_step_adv_model(x, u):
        z_pred = z_model(x)
        preds = adv_model(z_pred)
        val_loss_metric.update_state(u, preds)
        val_acc_metric.update_state(u, preds)

    best = float('inf')
    wait = 0
    
    fl = cfg['f_hidden_layers']
    fu = cfg['f_hidden_units']
    vl = cfg['v_hidden_layers']
    vu = cfg['v_hidden_units']
    al = cfg['a_hidden_layers']
    au = cfg['a_hidden_units']
    z_layer_f = cfg['z_layer_f']
    z_layer_a = cfg['z_layer_a']
    reg = cfg['reg']

    f_model = tf.keras.models.load_model(f'{exp_name}/saved_models/run_{run+1}/v_fair_fl_{fl}_fu_{fu}_vl_{vl}_vu_{vu}_z_{z_layer_f}_reg_{reg}_model.keras')
    z_model = create_z_model(f_model, z_layer_a)
    z_dim = f_model.layers[z_layer_a].output.shape[1]
    adv_model = create_v_model(al, au, z_dim, u_dim)
    
#     lr_schedule_a = tf.keras.optimizers.schedules.ExponentialDecay(cfg['initial_lr_a'],
#                                                                    decay_steps=10000,
#                                                                    decay_rate=0.98,
#                                                                    staircase=True)
    adv_optimizer = tf.keras.optimizers.SGD(cfg['initial_lr_a'])
    
    if u_dim == 1:
        trn_loss_metric = tf.keras.metrics.BinaryCrossentropy()
        val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
        trn_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    else:
        trn_loss_metric = tf.keras.metrics.CategoricalCrossentropy()
        val_loss_metric = tf.keras.metrics.CategoricalCrossentropy()
        trn_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    for epoch in range(cfg['epochs_a']):
        for (x, u, y) in trn:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)
            train_step_adv_model(x, u)
        trn_acc = trn_acc_metric.result()
        trn_loss = trn_loss_metric.result()
        trn_acc_metric.reset_state()
        trn_loss_metric.reset_state()

        for (x, u, y) in val:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)
            valid_step_adv_model(x, u)
        val_acc = val_acc_metric.result()
        val_loss = val_loss_metric.result()
        val_acc_metric.reset_state()
        val_loss_metric.reset_state()
        
        if print_results:
            print(f"Epoch: {epoch} -> training loss: {trn_loss.numpy():.3f}, valid loss: {val_loss.numpy():.3f}, training acc: {trn_acc.numpy():.3f}, valid acc: {val_acc.numpy():.3f}")

        wait += 1
        if val_loss < best-0.001:
            best = val_loss
            wait = 0
            print('Saving the best adversarial model...')
            adv_model.save(f'{exp_name}/saved_models/run_{run+1}/v_fair_al_{al}_au_{au}_fl_{fl}_fu_{fu}_vl_{vl}_vu_{vu}_zf_{z_layer_f}_za_{z_layer_a}_reg_{reg}_model.keras')
        if wait >= cfg['patience']:
            break
            
def save_results(ds, cfg, exp_name, n_runs, print_results=False):
    trn, val, tst = ds.get_dataset_in_tensor(batch_size=cfg['batch_size'])
    u_dim = trn.element_spec[1].shape[1]
    
    u = np.concatenate([u for _, u, _ in tst], axis=0)
    if u_dim > 1:
        u = np.argmax(u, axis=1)
    hu = entropy(u)
    
    fl = cfg['f_hidden_layers']
    fu = cfg['f_hidden_units']
    vl = cfg['v_hidden_layers']
    vu = cfg['v_hidden_units']
    al = cfg['a_hidden_layers']
    au = cfg['a_hidden_units']
    z_layer_f = cfg['z_layer_f']
    z_layer_a = cfg['z_layer_a']
    reg = cfg['reg']
    
    f_acc_list = []
    f_auc_list = []
    ddp_list = []
    deo_list = []
    deopp_list = []
    adv_acc_list = []
    adv_auc_list = []
    adv_info_list = []
    
    for run in range(n_runs):
        
        tf.keras.utils.set_random_seed(run)
    
        f_model = tf.keras.models.load_model(f'{exp_name}/saved_models/run_{run+1}/v_fair_fl_{fl}_fu_{fu}_vl_{vl}_vu_{vu}_z_{z_layer_f}_reg_{reg}_model.keras')
        z_model = create_z_model(f_model, z_layer_a)
        adv_model = tf.keras.models.load_model(f'{exp_name}/saved_models/run_{run+1}/v_fair_al_{al}_au_{au}_fl_{fl}_fu_{fu}_vl_{vl}_vu_{vu}_zf_{z_layer_f}_za_{z_layer_a}_reg_{reg}_model.keras')

        y_pred_list = []
        u_pred_list = []
        y_list = []
        u_list = []
        total_loss = 0.
        for (x, u, y) in tst:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)
            preds_y = f_model(x)
            y_pred = tf.cast(preds_y > 0.5, dtype=tf.float32)
            y_pred_list.append(y_pred.numpy())
            y_list.append(y.numpy())
            z_pred = z_model(x)
            preds_u = adv_model(z_pred)
            if u_dim == 1:
                loss = BCE_loss(u, preds_u)
            else:
                loss = CCE_loss(u, preds_u)
            total_loss += loss
            if u_dim == 1:
                u_pred = tf.cast(preds_u > 0.5, dtype=tf.float32)
            else:
                u_pred = preds_u
            u_pred_list.append(u_pred.numpy())
            u_list.append(u.numpy())
        y_pred = np.concatenate(y_pred_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        u_pred = np.concatenate(u_pred_list, axis=0)
        u = np.concatenate(u_list, axis=0)

        f_acc_list.append(accuracy(y,y_pred)*100)
        f_auc_list.append(auc_score(y,y_pred))
        adv_auc_list.append(auc_score(u,u_pred))
        if u_dim == 1:
            ddp_list.append(ddp(y_pred,u))
            deo_list.append(deo(y,y_pred,u))
            deopp_list.append(deopp(y,y_pred,u))
            adv_acc_list.append(accuracy(u,u_pred)*100)
        else:
            u = np.argmax(u, axis=1)
            ddp_list.append(ddp(y_pred,u))
            deo_list.append(deo(y,y_pred,u))
            deopp_list.append(deopp(y,y_pred,u))
            u_pred = np.argmax(u_pred, axis=1)
            adv_acc_list.append(accuracy(u,u_pred)*100)
        adv_info_list.append(hu-total_loss.numpy()/len(tst))
    
    if print_results:
        print(f'Classifier Test accuracy: {np.mean(f_acc_list):.2f} ({np.std(f_acc_list):.2f})')
        print(f'Classifier AUC score: {np.mean(f_auc_list):.3f} ({np.std(f_auc_list):.3f})')
        print(f'Demographic parity difference: {np.mean(ddp_list):.3f} ({np.std(ddp_list):.3f})')
        print(f'Equalized odds difference: {np.mean(deo_list):.3f} ({np.std(deo_list):.3f})')
        print(f'Equalized opportunity difference: {np.mean(deopp_list):.3f} ({np.std(deopp_list):.3f})')
        print(f'Adversary accuracy: {np.mean(adv_acc_list):.2f} ({np.std(adv_acc_list):.2f})')
        print(f'Adversary AUC score: {np.mean(adv_auc_list):.3f} ({np.std(adv_auc_list):.3f})')
        print(f'V-info (z->u): {np.mean(adv_info_list):.3f} ({np.std(adv_info_list):.3f})')
                                             
    results = {}
    results['f_acc'] = f_acc_list
    results['f_auc'] = f_auc_list
    results['ddp'] = ddp_list
    results['deo'] = deo_list
    results['deopp'] = deopp_list
    results['adv_acc'] = adv_acc_list
    results['adv_auc'] = adv_auc_list
    results['adv_info'] = adv_info_list

    results_dir = f'{exp_name}/saved_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(f'{results_dir}/v_fair_al_{al}_au_{au}_fl_{fl}_fu_{fu}_vl_{vl}_vu_{vu}_zf_{z_layer_f}_za_{z_layer_a}_reg_{reg}_results', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
