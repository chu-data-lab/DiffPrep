import torch
from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy

def make_batch(X, y, batch_size, shuffle=False):
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    start_idx = 0

    while True:
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]

        start_idx += batch_size
        if start_idx >= X.shape[0]:
            break

class BaselineTrainerSGD(object):
    def __init__(self, model, loss_fn, model_optimizer, model_scheduler, params, writer=None):
        self.model = model
        self.loss_fn = loss_fn
        self.model_optimizer = model_optimizer
        self.params = params
        self.device = self.params["device"]
        self.writer = None
        self.model_scheduler = model_scheduler
        self.writer = writer

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, verbose=True):
        best_val_loss = float("inf")
        best_model = None
        best_result = None
        
        # for early stopping
        last_best_val_acc = float("-inf") 
        patience = self.params["patience"]

        # if self.writer is not None and self.pipeline is not None:
        #     self.writer.add_baseline_pipeline(self.pipeline.pipeline, global_step=0)

        e = 0
        if verbose:
            pbar = tqdm(total=self.params["num_epochs"])

        # start training
        while e < self.params["num_epochs"]:
            # print("epoch:", e)
            tic = time.time()
            tr_loss, tr_acc = self.train(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            test_loss, test_acc = self.evaluate(X_test, y_test)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_result = {
                    "best_epoch": e,
                    "best_val_loss": val_loss,
                    "best_tr_acc": tr_acc,
                    "best_val_acc": val_acc,
                    "best_test_acc": test_acc,
                }
                best_model = deepcopy(self.model.state_dict())

            model_lr = self.model_optimizer.param_groups[0]['lr']
            # t.set_postfix(tr_loss=tr_loss, val_loss=val_loss)
            # print("tr loss:", tr_loss, "tr_acc", tr_acc, "val_loss", val_loss, "val_acc", val_acc)

            if self.model_scheduler is not None:
                self.model_scheduler.step(val_loss)

            if self.writer is not None:
                self.writer.add_scalar('tr_loss', tr_loss, global_step=e)
                self.writer.add_scalar('tr_acc', tr_acc, global_step=e)
                self.writer.add_scalar('val_loss', val_loss, global_step=e)
                self.writer.add_scalar('val_acc', val_acc, global_step=e)
                self.writer.add_scalar('test_loss', test_loss, global_step=e)
                self.writer.add_scalar('test_acc', test_acc, global_step=e)
                self.writer.add_scalar('model_lr', model_lr, global_step=e)

            # check stopping criteria every 100 epoch
            if e % 100 == 0:
                if best_result["best_val_acc"] - last_best_val_acc < 0.001:
                    patience = patience - 1
                else:
                    last_best_val_acc = best_result["best_val_acc"]
                    patience = self.params["patience"]
            
            if patience <= 0:
                break

            e += 1     
            
            if verbose:
                pbar.update(1)
        
        if verbose:
            pbar.close()
            
        # print("Finished", best_result)
        return best_result, best_model

    def train(self, X_train, y_train):
        self.model.train()
        
        train_iter =  make_batch(X_train, y_train, self.params["batch_size"])
    
        tr_correct = 0
        tr_loss = 0
        n_batches = 0
        for i, (X_train_batch, y_train_batch) in enumerate(train_iter):
            output_train = self.model(X_train_batch.to(self.device))
            loss_train = self.loss_fn(output_train, y_train_batch.to(self.device))

            self.model_optimizer.zero_grad()
            loss_train.backward()
            self.model_optimizer.step()

            _, preds = torch.max(output_train, 1)
            correct = torch.sum(preds.cpu() == y_train_batch)

            tr_loss += loss_train.item()
            tr_correct += correct.item()
            n_batches += 1

        tr_acc = tr_correct / len(y_train)
        tr_loss = tr_loss / n_batches
        return tr_loss, tr_acc

    def evaluate(self, X, y):
        self.model.eval()
        output = self.model(X.to(self.device))
        loss = self.loss_fn(output, y.to(self.device))
        _, preds = torch.max(output, 1)
        correct = torch.sum(preds.cpu() == y)
        acc = correct.item() / len(y)
        return loss.item(), acc

