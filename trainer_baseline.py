import tensorflow as tf
import numpy as np
import os
import pickle
from model import create_f_model, create_z_model, create_v_model
from utils import *

def train_baseline_model_f(ds, cfg, exp_name, run, print_results=False):
    
    tf.keras.utils.set_random_seed(run)
    
    trn, val, tst = ds.get_dataset_in_tensor(batch_size=cfg['batch_size'])
    if cfg['include_u']:
        x_dim = trn.element_spec[0].shape[1]+trn.element_spec[1].shape[1]
    else:
        x_dim = trn.element_spec[0].shape[1]
    u_dim = trn.element_spec[1].shape[1]
    y_dim = trn.element_spec[2].shape[1]
    
    @tf.function
    def train_step_f_model(x, y):
        with tf.GradientTape() as f_tape:
            preds = f_model(x)
            f_loss = BCE_loss(y, preds)

        gradients_of_f = f_tape.gradient(f_loss, f_model.trainable_variables)
        f_optimizer.apply_gradients(zip(gradients_of_f, f_model.trainable_variables))
        trn_loss_metric.update_state(y, preds)
        y_pred = tf.cast(preds > 0.5, dtype=tf.float32)
        trn_acc_metric.update_state(y, y_pred)

    @tf.function
    def valid_step_f_model(x, y):
        preds = f_model(x)
        val_loss_metric.update_state(y, preds)
        y_pred = tf.cast(preds > 0.5, dtype=tf.float32)
        val_acc_metric.update_state(y, y_pred)
        
    best = 0.0
    wait = 0
    
    fl = cfg['f_hidden_layers']
    fu = cfg['f_hidden_units']
    
    f_model = create_f_model(fl, fu, x_dim, y_dim)
    
    initial_lr = cfg['initial_lr_f']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr,
                                                                 decay_steps=1000,
                                                                 decay_rate=0.98,
                                                                 staircase=True)
    f_optimizer = tf.keras.optimizers.Adam(lr_schedule)
    
    trn_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    trn_acc_metric = tf.keras.metrics.Accuracy()
    val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    val_acc_metric = tf.keras.metrics.Accuracy()
    
    for epoch in range(cfg['epochs']):
        for (x, u, y) in trn:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)
            train_step_f_model(x, y)
        trn_acc = trn_acc_metric.result()
        trn_loss = trn_loss_metric.result()
        trn_acc_metric.reset_state()
        trn_loss_metric.reset_state()

        for (x, u, y) in val:
            x, u, y = tf.convert_to_tensor(x), tf.convert_to_tensor(u), tf.convert_to_tensor(y)
            if cfg['include_u']:
                x = tf.concat([x, u], axis=1)
            valid_step_f_model(x, y)
        val_acc = val_acc_metric.result()
        val_loss = val_loss_metric.result()
        val_acc_metric.reset_state()
        val_loss_metric.reset_state()
        
        if print_results:
            print(f"Epoch: {epoch} -> training loss: {trn_loss.numpy():.3f}, valid loss: {val_loss.numpy():.3f}, training acc: {trn_acc.numpy():.3f}, valid acc: {val_acc.numpy():.3f}")
        
        if not os.path.exists(f'{exp_name}/saved_models/run_{run+1}'):
            os.makedirs(f'{exp_name}/saved_models/run_{run+1}')
        
        wait += 1
        if val_acc > best+0.001:
            best = val_acc
            wait = 0
            f_model.save(f'{exp_name}/saved_models/run_{run+1}/baseline_fl_{fl}_fu_{fu}_model.keras')
        if wait >= cfg['patience']:
            break
            
def train_baseline_model_adv(ds, cfg, exp_name, run, print_results=False):
    
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
        if u_dim == 1:
            u_pred = tf.cast(preds > 0.5, dtype=tf.float32)
            trn_acc_metric.update_state(u, u_pred)
        else:
            trn_acc_metric.update_state(u, preds)
        

    @tf.function
    def valid_step_adv_model(x, u):
        z_pred = z_model(x)
        preds = adv_model(z_pred)
        val_loss_metric.update_state(u, preds)
        if u_dim == 1:
            u_pred = tf.cast(preds > 0.5, dtype=tf.float32)
            val_acc_metric.update_state(u, u_pred)
        else:
            val_acc_metric.update_state(u, preds)

    best = float('inf')
    wait = 0
    
    fl = cfg['f_hidden_layers']
    fu = cfg['f_hidden_units']
    al = cfg['a_hidden_layers']
    au = cfg['a_hidden_units']
    z_layer = cfg['z_layer']

    f_model = tf.keras.models.load_model(f'{exp_name}/saved_models/run_{run+1}/baseline_fl_{fl}_fu_{fu}_model.keras')
    z_model = create_z_model(f_model, z_layer)
    z_dim = f_model.layers[z_layer].output.shape[1]
    adv_model = create_v_model(al, au, z_dim, u_dim)
    
    initial_lr = cfg['initial_lr_a']
    adv_optimizer = tf.keras.optimizers.SGD(initial_lr)
    
    if u_dim == 1:
        trn_loss_metric = tf.keras.metrics.BinaryCrossentropy()
        val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
        trn_acc_metric = tf.keras.metrics.Accuracy()
        val_acc_metric = tf.keras.metrics.Accuracy()
    else:
        trn_loss_metric = tf.keras.metrics.CategoricalCrossentropy()
        val_loss_metric = tf.keras.metrics.CategoricalCrossentropy()
        trn_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    

    for epoch in range(cfg['epochs']):
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
            adv_model.save(f'{exp_name}/saved_models/run_{run+1}/baseline_al_{al}_au_{au}_fl_{fl}_fu_{fu}_z_{z_layer}_model.keras')
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
    al = cfg['a_hidden_layers']
    au = cfg['a_hidden_units']
    z_layer = cfg['z_layer']
    
    f_acc_list = []
    f_auc_list = []
    dp_list = []
    eo_list = []
    eopp_list = []
    adv_acc_list = []
    adv_auc_list = []
    adv_info_list = []
    
    for run in range(n_runs):
        
        tf.keras.utils.set_random_seed(run)
    
        f_model = tf.keras.models.load_model(f'{exp_name}/saved_models/run_{run+1}/baseline_fl_{fl}_fu_{fu}_model.keras')
        z_model = create_z_model(f_model, z_layer)
        adv_model = tf.keras.models.load_model(f'{exp_name}/saved_models/run_{run+1}/baseline_al_{al}_au_{au}_fl_{fl}_fu_{fu}_z_{z_layer}_model.keras')

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
            dp_list.append(ddp(y_pred,u))
            eo_list.append(deo(y,y_pred,u))
            eopp_list.append(deopp(y,y_pred,u))
            adv_acc_list.append(accuracy(u,u_pred)*100)
        else:
            u = np.argmax(u, axis=1)
            dp_list.append(ddp(y_pred,u))
            eo_list.append(deo(y,y_pred,u))
            eopp_list.append(deopp(y,y_pred,u))
            u_pred = np.argmax(u_pred, axis=1)
            adv_acc_list.append(accuracy(u,u_pred)*100)
        adv_info_list.append(hu-total_loss.numpy()/len(tst))
    
    if print_results:
        print(f'Classifier Test accuracy: {np.mean(f_acc_list):.2f} ({np.std(f_acc_list):.2f})')
        print(f'Classifier AUC score: {np.mean(f_auc_list):.3f} ({np.std(f_auc_list):.3f})')
        print(f'Demographic parity difference: {np.mean(dp_list):.3f} ({np.std(dp_list):.3f})')
        print(f'Equalized odds difference: {np.mean(eo_list):.3f} ({np.std(eo_list):.3f})')
        print(f'Equalized opportunity difference: {np.mean(eopp_list):.3f} ({np.std(eopp_list):.3f})')
        print(f'Adversary accuracy: {np.mean(adv_acc_list):.2f} ({np.std(adv_acc_list):.2f})')
        print(f'Adversary AUC score: {np.mean(adv_auc_list):.3f} ({np.std(adv_auc_list):.3f})')
        print(f'V-info (z->u): {np.mean(adv_info_list):.3f} ({np.std(adv_info_list):.3f})')
                                             
    results = {}
    results['f_acc'] = f_acc_list
    results['f_auc'] = f_auc_list
    results['ddp'] = dp_list
    results['deo'] = eo_list
    results['deopp'] = eopp_list
    results['adv_acc'] = adv_acc_list
    results['adv_auc'] = adv_auc_list
    results['adv_info'] = adv_info_list

    results_dir = f'{exp_name}/saved_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(f'{results_dir}/baseline_al_{al}_au_{au}_fl_{fl}_fu_{fu}_z_{z_layer}_results', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)