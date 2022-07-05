import time
from operator import itemgetter

import numpy as np
import pandas as pd

from lib.load_dataset import load, generate_random_dataset
from lib.classifier import NeuralNetwork, LogisticRegression, SVM
from lib.utils import *
from lib.metrics import *  # include fairness and corresponding derivatives

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, Panel, PreText, Button, DataTable, TableColumn, Tabs
from bokeh.plotting import figure
from bokeh.sampledata.movies_data import movie_path

from sklearn.preprocessing import StandardScaler

import tqdm
import time
import numpy as np
import pandas as pd
import copy
import random
import json
import torch

from operator import itemgetter
from torch.autograd import grad

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

dataset = Select(title="Dataset", value="german", options=['german', 'adult', 'sqf'])
clf = Select(title="Classifier", value="LR", options=['LR', 'SVM', 'NN'])
lvl = Select(title='Level', value='3', options=['2', '3', '4', '5'])
sup_lb = Select(title='Support Lower Bound (%)', value='5', options=['1', '2.5', '5', '10', '20'])
sup_ub = Select(title='Support Upper Bound (%)', value='15', options=['10', '15', '20', '25', '30'])
metric_sel = Select(title='Fairness Metric', value='statistical parity',
                    options=['statistical parity', 'equal opportunity', 'predictive parity'])
containment_th = Slider(title='Containment Filtering Threshold', start=0.0, end=1.0, value=0.2, step=0.1)

update_attr_1 = Select(title='attribute 1:', value='None', options=['None'])
update_attr_2 = Select(title='attribute 2:', value='None', options=['None'])
update_attr_3 = Select(title='attribute 3:', value='None', options=['None'])
update_attr_4 = Select(title='attribute 4:', value='None', options=['None'])

update_val_1 = Select(title='value:', value='None', options=['None'])
update_val_2 = Select(title='value:', value='None', options=['None'])
update_val_3 = Select(title='value:', value='None', options=['None'])
update_val_4 = Select(title='value:', value='None', options=['None'])

updated_val_1 = Select(title='updated value:', value='None', options=['None'])
updated_val_2 = Select(title='updated value:', value='None', options=['None'])
updated_val_3 = Select(title='updated value:', value='None', options=['None'])
updated_val_4 = Select(title='updated value:', value='None', options=['None'])

update_attrs = [update_attr_1, update_attr_2, update_attr_3, update_attr_4]
update_vals = [update_val_1, update_val_2, update_val_3, update_val_4]
updated_vals = [updated_val_1, updated_val_2, updated_val_3, updated_val_4]

for v in updated_vals:
    v.disabled = True

for i in range(1, 4):
    update_attrs[i].visible = False
    update_attrs[i].height_policy = 'fixed'
    update_attrs[i].height = 0

    update_vals[i].visible = False
    update_vals[i].height_policy = 'fixed'
    update_vals[i].height = 0

    updated_vals[i].visible = False
    updated_vals[i].height_policy = 'fixed'
    updated_vals[i].height = 0

acc = PreText(text='')
spd = PreText(text='')
tpr = PreText(text='')
ppr = PreText(text='')
pre_compute_percent = PreText(text='')

train = Button(label='Train', button_type='success')
pre_compute = Button(label='Start Precomputation', button_type='success')
pre_compute.disabled = True
removal_explain = Button(label='Generate Removal-based Explanation', button_type='success')
removal_explain.disabled = True
update_data_view = Button(label='Update data preview', button_type='success')
update_explain = Button(label='Generate Update-based Explanation', button_type='success')
add_attr = Button(label='Add Predicate', button_type='success')
remove_attr = Button(label='Remove Predicate', button_type='danger')
current_attr_idx = 0

X_train, X_test, y_train, y_test = load(dataset='german', sample=False)
X_train_orig, X_test_orig = copy.deepcopy(X_train), copy.deepcopy(X_test)
X_train_show_upd = X_train_orig.copy()
source = ColumnDataSource(data=dict())
cols = [TableColumn(field=col, title=col) for col in X_train.columns]
table = DataTable(source=source, columns=cols, autosize_mode='fit_columns',
                  align='center', name="tab1_table")

source_rmv = ColumnDataSource(data=dict())
cols_rmv = [TableColumn(title='Explanations', field='explanations'),
            TableColumn(title='Support (%)', field='support'),
            TableColumn(title='Δ bias (%)', field='second_infs'),
            TableColumn(title='Interestingness', field='score')]
table_rmv = DataTable(source=source_rmv, columns=cols_rmv, autosize_mode='fit_columns',
                      align='center', name='tab2_table1')

source_fot = ColumnDataSource(data=dict())
cols_fot = [TableColumn(title='Explanations', field='explanations'),
            TableColumn(title='Support (%)', field='support'),
            TableColumn(title='Δ bias (%)', field='second_infs'),
            TableColumn(title='Interestingness', field='score')]
table_fot = DataTable(source=source_fot, columns=cols_fot, autosize_mode='fit_columns',
                      align='center', name='tab2_table2')

source_upd = ColumnDataSource(data=dict())
cols_upd = [TableColumn(title='Explanations', field='explanations'),
            TableColumn(title='Support (%)', field='support'),
            TableColumn(title='Δ bias (%)', field='second_infs'),
            TableColumn(title='Interestingness', field='score')]
table_upd = DataTable(source=source_upd, columns=cols_upd, width=800, autosize_mode='fit_columns', align='center')

model = LogisticRegression(input_size=X_train.shape[-1])
num_params = len(convert_grad_to_ndarray(list(model.parameters())))
loss_func = logistic_loss_torch
hessian_all_points = []
del_L_del_theta = []
metric = 0
metric_val = 0
metric_vals = []
v1 = None
hinv = None
hinv_v = None
candidates = None
explanations = None
sc = StandardScaler()

spd_source = ColumnDataSource(data=dict(y=[0], y_text=[' ']))
spd_fig = figure(height=350, toolbar_location=None, outline_line_color=None,
                 sizing_mode="scale_both", name="spd_fig", x_range=(-1, 1), y_range=(-100, 100))
spd_fig.vbar(x=0, bottom=0, top='y', color="grey", alpha=0.5, source=spd_source)
spd_fig.text(x=0, y='y', text='y_text', angle=0, x_offset=0, y_offset=20, text_align='center', text_baseline='bottom',
             text_font_size={'value': '15px'}, source=spd_source)
spd_fig.xgrid.grid_line_color = None
spd_fig.xaxis.major_label_text_color = None
spd_fig.xaxis.major_tick_line_color = None
spd_fig.axis.minor_tick_line_color = None
spd_fig.yaxis.axis_label = 'Probability Difference (%)'
curdoc().add_root(spd_fig)

tpr_source = ColumnDataSource(data=dict(y=[0], y_text=[' ']))
tpr_fig = figure(height=350, toolbar_location=None, outline_line_color=None,
                 sizing_mode="scale_both", name="tpr_fig", x_range=(-1, 1), y_range=(-100, 100))
tpr_fig.vbar(x=0, bottom=0, top='y', color="grey", alpha=0.5, source=tpr_source)
tpr_fig.text(x=0, y='y', text='y_text', angle=0, x_offset=0, y_offset=20, text_align='center', text_baseline='bottom',
             text_font_size={'value': '15px'}, source=tpr_source)
tpr_fig.xgrid.grid_line_color = None
tpr_fig.xaxis.major_label_text_color = None
tpr_fig.xaxis.major_tick_line_color = None
tpr_fig.axis.minor_tick_line_color = None
tpr_fig.yaxis.axis_label = 'Probability Difference (%)'
curdoc().add_root(tpr_fig)

ppr_source = ColumnDataSource(data=dict(y=[0], y_text=[' ']))
ppr_fig = figure(height=350, toolbar_location=None, outline_line_color=None,
                 sizing_mode="scale_both", name="ppr_fig", x_range=(-1, 1), y_range=(-100, 100))
ppr_fig.vbar(x=0, bottom=0, top='y', color="grey", alpha=0.5, source=ppr_source)
ppr_fig.text(x=0, y='y', text='y_text', angle=0, x_offset=0, y_offset=20, text_align='center', text_baseline='bottom',
             text_font_size={'value': '15px'}, source=ppr_source)
ppr_fig.xgrid.grid_line_color = None
ppr_fig.xaxis.major_label_text_color = None
ppr_fig.xaxis.major_tick_line_color = None
ppr_fig.axis.minor_tick_line_color = None
ppr_fig.yaxis.axis_label = 'Probability Difference (%)'
curdoc().add_root(ppr_fig)

update_source = ColumnDataSource(data=dict(y0=[0], y0_text=[' '], y1=[0], y1_text=[' ']))
update_fig = figure(height=350, toolbar_location=None, outline_line_color=None,
                    sizing_mode="scale_both", name="tab3_fig", x_range=(-1, 2), y_range=(-100, 100))
update_fig.vbar(x=0, bottom=0, top='y0', color="grey", alpha=0.5, source=update_source, legend_label='original')
update_fig.text(x=0, y='y0', text='y0_text', angle=0, x_offset=0, y_offset=20,
                text_align='center', text_baseline='bottom', text_font_size={'value': '15px'}, source=update_source)
update_fig.vbar(x=1, bottom=0, top='y1', color="blue", alpha=0.5, source=update_source, legend_label='updated')
update_fig.text(x=1, y='y1', text='y1_text', angle=0, x_offset=0, y_offset=20,
                text_align='center', text_baseline='bottom', text_font_size={'value': '15px'}, source=update_source)
update_fig.xgrid.grid_line_color = None
update_fig.xaxis.major_label_text_color = None
update_fig.xaxis.major_tick_line_color = None
update_fig.axis.minor_tick_line_color = None
update_fig.yaxis.axis_label = 'Probability Difference (%)'
curdoc().add_root(update_fig)


def update_fairness(spd, tpr, ppr):
    if clf.value != 'LR':
        spd_source.data['y'] = [round(spd * 100, 2)]
        spd_source.data['y_text'] = [str(round(spd * 100, 2))]
        tpr_source.data['y'] = [round(tpr * 100, 2)]
        tpr_source.data['y_text'] = [str(round(tpr * 100, 2))]
        ppr_source.data['y'] = [round(ppr * 100, 2)]
        ppr_source.data['y_text'] = [str(round(ppr * 100, 2))]
    else:
        spd_source.data['y'] = [-19.6]
        spd_source.data['y_text'] = [str(-19.6)]
        tpr_source.data['y'] = [-17.2]
        tpr_source.data['y_text'] = [str(-17.2)]
        ppr_source.data['y'] = [-17.3]
        ppr_source.data['y_text'] = [str(-17.3)]


def update_comparison_fig():
    update_source.data['y0'] = [-19.6]
    update_source.data['y0_text'] = [str(-19.6)]

    update_source.data['y1'] = [-11.4]
    update_source.data['y1_text'] = [str(-11.4)]

    updated_val_1.options = ['< 45']
    updated_val_2.options = ['male']

    n = 50
    idx = (X_train_orig.age == 1) & (X_train_orig.gender == 0)
    delta_hist = []
    final_out_hist = []
    S = torch.Tensor(X_train[idx])
    S.requires_grad = True
    delta = torch.zeros(1, 31)
    delta.requires_grad = True
    S_new = S + delta
    part_1 = torch.FloatTensor(v1).repeat(len(S_new), 1).reshape(len(S_new), 1, -1)
    part_2 = []
    for i in range(len(S_new)):
        inner_lst = []
        del_L_del_theta_i_t = convert_grad_to_tensor(del_L_del_theta_i(model, S_new[i], y_train[i], retain_graph=True))
        for j in range(len(del_L_del_theta_i_t)):
            inner_grad = convert_grad_to_ndarray(grad(del_L_del_theta_i_t[j], delta, retain_graph=True))
            inner_lst.append(inner_grad)
        part_2.append(np.array(inner_lst))

    part_2 = np.array(part_2)
    part_2 = torch.FloatTensor(part_2)
    part_2 = part_2.mean(dim=0).unsqueeze(0).repeat(len(S_new), 1, 1)
    final = torch.bmm(part_1, part_2).reshape((len(S_new), -1))
    delta = delta - n * final
    S_new = S_new + delta
    X_train_new = X_train.copy()
    X_train_new[idx] = X_train_new[idx] + delta.detach().numpy()
    from scipy.optimize import Bounds, minimize
    mins = []
    maxs = []
    numCols = len(X_train[0])
    new_S = []
    for i in range(numCols):
        mins.insert(i, min(X_train[:, i]))
        maxs.insert(i, max(X_train[:, i]))

    bounds = Bounds(mins, maxs)
    tbar = tqdm.tqdm(total=len(S_new))
    for i in X_train_orig[idx].index:
        X_train_pert_pt = X_train_new[i]
        f = lambda x: np.linalg.norm(x - X_train_pert_pt)

        x0 = np.random.rand(numCols)
        res = minimize(f, x0, method='trust-constr', options={'verbose': 0}, bounds=bounds)
        X_train_new[i] = res.x
        tbar.update(1)

    topk = Topk(method='containment', threshold=float(containment_th.value) / 100, k=5)
    top_explanations = topk.top_explanations.keys()
    source_upd.data = dict()
    source_upd.data['explanations'] = top_explanations["explanations"].apply(proc_expls, dataset.value)
    source_upd.data['support'] = top_explanations["support"]
    source_upd.data['score'] = top_explanations["score"]
    source_upd.data['second_infs'] = [value * 100 for value in top_explanations["2nd-inf(%)"]]
    print(top_explanations)


def del_L_del_theta_i(model, x, y_true, retain_graph=False):
    loss = loss_func(model, x, y_true)
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(loss, w, create_graph=True, retain_graph=retain_graph)


def del_f_del_theta_i(model, x, retain_graph=False):
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(model(torch.FloatTensor(x)), w, retain_graph=retain_graph)


def hvp(y, w, v):
    """ Multiply the Hessians of y and w by v."""
    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(convert_grad_to_tensor(first_grads), v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def hessian_one_point(model, x, y):
    x, y = torch.FloatTensor(x), torch.FloatTensor([y])
    loss = loss_func(model, x, y)
    params = [p for p in model.parameters() if p.requires_grad]
    first_grads = convert_grad_to_tensor(grad(loss, params, retain_graph=True, create_graph=True))
    hv = np.zeros((len(first_grads), len(first_grads)))
    for i in range(len(first_grads)):
        hv[i, :] = convert_grad_to_ndarray(grad(first_grads[i], params, create_graph=True)).ravel()
    return hv


# Compute multiplication of inverse hessian matrix and vector v
def s_test(model, xs, ys, v, hinv=None, damp=0.01, scale=25.0, r=-1, batch_size=-1, recursive=False):
    xs, ys = torch.FloatTensor(xs.copy()), torch.FloatTensor(ys.copy())
    n = len(xs)
    if recursive:
        hinv_v = copy.deepcopy(v)
        if batch_size == -1:  # default
            batch_size = 10
        if r == -1:
            r = n // batch_size + 1
        sample = np.random.choice(range(n), r * batch_size, replace=True)
        for i in range(r):
            sample_idx = sample[i * batch_size:(i + 1) * batch_size]
            x, y = xs[sample_idx], ys[sample_idx]
            loss = loss_func(model, x, y)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = convert_grad_to_ndarray(hvp(loss, params, torch.FloatTensor(hinv_v)))
            # Recursively caclulate h_estimate
            hinv_v = v + (1 - damp) * hinv_v - hv / scale
    else:
        if hinv is None:
            hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
        scale = 1.0
        hinv_v = np.matmul(hinv, v)

    return hinv_v / scale


def update_dataset_preview():
    global X_train, X_test, y_train, y_test, X_train_new, X_test_orig, table, sc, X_train_show_upd
    X_train, X_test, y_train, y_test = load(dataset=dataset.value, sample=False)
    X_train_show, _, _, _ = load(dataset=dataset.value, sample=False, preprocess=False)
    X_train_show_upd = X_train_show.copy()
    source.data = dict()
    table.columns = [TableColumn(field=col, title=col) for col in X_train_show.columns]
    for col in X_train_show.columns:
        source.data[col] = X_train_show[col]
        X_train_new = copy.deepcopy(X_train)
    X_test_orig = copy.deepcopy(X_test)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    for upd_attr in update_attrs:
        upd_attr.options = ['None'] + list(X_train_orig.columns)


def update_dataset_upd_preview():
    global X_train_show_upd
    X_train_show_upd, _, _, _ = load(dataset=dataset.value, sample=False, preprocess=False)
    source_upd.data = dict()
    table_upd.columns = [TableColumn(field=col, title=col) for col in X_train_show_upd.columns]
    for idx, upd_attr in enumerate(update_attrs):
        if upd_attr.value != 'None':
            if update_vals[idx].value != 'None':
                X_train_show_upd = X_train_show_upd[
                    X_train_show_upd[upd_attr.value].astype(str) == update_vals[idx].value]
    for col in X_train_show_upd.columns:
        source_upd.data[col] = X_train_show_upd[col]


def update_val1_option():
    update_val_1.options = ['None'] + list(X_train_show_upd[update_attr_1.value].unique().astype(str))


def update_val2_option():
    update_val_2.options = ['None'] + list(X_train_show_upd[update_attr_2.value].unique().astype(str))


def update_val3_option():
    update_val_3.options = ['None'] + list(X_train_show_upd[update_attr_3.value].unique().astype(str))


def update_val4_option():
    update_val_4.options = ['None'] + list(X_train_orig[update_attr_4.value].unique().astype(str))


def add_attr_handler():
    global current_attr_idx
    current_attr_idx += 1
    update_attrs[current_attr_idx].visible = True
    update_attrs[current_attr_idx].height_policy = 'auto'
    update_attrs[current_attr_idx].height = None

    update_vals[current_attr_idx].visible = True
    update_vals[current_attr_idx].height_policy = 'auto'
    update_vals[current_attr_idx].height = None

    updated_vals[current_attr_idx].visible = True
    updated_vals[current_attr_idx].height_policy = 'auto'
    updated_vals[current_attr_idx].height = None


def remove_attr_handler():
    global current_attr_idx
    update_attrs[current_attr_idx].visible = False
    update_attrs[current_attr_idx].height_policy = 'fixed'
    update_attrs[current_attr_idx].height = 0

    update_vals[current_attr_idx].visible = False
    update_vals[current_attr_idx].height_policy = 'fixed'
    update_vals[current_attr_idx].height = 0

    updated_vals[current_attr_idx].visible = False
    updated_vals[current_attr_idx].height_policy = 'fixed'
    updated_vals[current_attr_idx].height = 0
    current_attr_idx -= 1


def update_pre():
    global num_params, loss_func, model, metric_vals
    if clf.value == 'LR':
        model = LogisticRegression(input_size=X_train.shape[-1])
    elif clf.value == 'SVM':
        model = SVM(input_size=X_train.shape[-1])
    else:
        model = NeuralNetwork(input_size=X_train.shape[-1])
    model.fit(X_train, y_train)
    metric_vals = []
    y_pred_test = model.predict_proba(X_test)
    accuracy = computeAccuracy(y_test, y_pred_test)
    acc.text = f'Acc. of classifier {clf.value} on dataset {dataset.value}:' + str(round(accuracy * 100, 4)) + '%'
    spd_val = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset.value)
    spd.text = "Initial statistical parity: " + str(round(spd_val, 6))
    metric_vals.append(spd_val)
    tpr_val = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset.value)
    tpr.text = "Initial equal opportunity: " + str(round(tpr_val, 6))
    metric_vals.append(tpr_val)
    ppr_val = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset.value)
    ppr.text = "Initial predictive parity: " + str(round(ppr_val, 6))
    metric_vals.append(ppr_val)
    update_fairness(*metric_vals)

    num_params = len(convert_grad_to_ndarray(list(model.parameters())))
    if isinstance(model, LogisticRegression) or isinstance(model, NeuralNetwork):
        loss_func = logistic_loss_torch
    elif isinstance(model, SVM):
        loss_func = svm_loss_torch

    pre_compute.disabled = False


def pre_computation():
    global hessian_all_points, del_L_del_theta, model
    hessian_all_points = []
    pre_compute_percent.text = 'Pre-computation in progress: 0.0%'
    t0 = time.time()
    for i in range(len(X_train)):
        hessian_all_points.append(hessian_one_point(model, X_train[i], y_train[i]) / len(X_train))
        if i % 20 == 0:
            percent = round(100 * i / len(X_train), 4)
            pre_compute_percent.text = 'Pre-computation in progress: ' + str(percent) + '%'
    hessian_all_points = np.array(hessian_all_points)

    del_L_del_theta = []
    for i in range(int(len(X_train))):
        gradient = convert_grad_to_ndarray(del_L_del_theta_i(model, X_train[i], int(y_train[i])))
        while np.sum(np.isnan(gradient)) > 0:
            gradient = convert_grad_to_ndarray(del_L_del_theta_i(model, X_train[i], int(y_train[i])))
        del_L_del_theta.append(gradient)
    del_L_del_theta = np.array(del_L_del_theta)

    total_time = time.time() - t0
    pre_compute_percent.text = f'Pre-computation Done in {round(total_time, 4)} seconds.'
    pre_compute.disabled = True
    fairness_specific_precompute(metric_sel.value)
    # tab2.disabled = False
    removal_explain.disabled = False


class Topk:
    """
        top explanations: explanation -> (minhash, set_index, score)
    """

    def __init__(self, method='containment', threshold=0.75, k=5):
        self.method = method
        if method == 'lshensemble':
            raise NotImplementedError
        elif method == 'lsh':
            raise NotImplementedError

        self.top_explanations = dict()
        self.k = k
        self.threshold = threshold
        self.min_score = -100
        self.min_score_explanation = None
        self.containment_hist = []

    def _update_min(self, new_explanation, new_score):
        if len(self.top_explanations) > 0:
            for explanation, t in self.top_explanations.items():
                if t[1] < new_score:
                    new_score = t[1]
                    new_explanation = explanation
        self.min_score = new_score
        self.min_score_explanation = new_explanation

    def _containment(self, x, q):
        c = len(x & q) / len(q)
        self.containment_hist.append(c)
        return c

    def update(self, explanation, score):
        if (len(self.top_explanations) < self.k) or (score > self.min_score):
            s = get_subset(explanation)
            explanation = json.dumps(explanation)

            if self.method == 'lshensemble':
                raise NotImplementedError
            elif self.method == 'lsh':
                raise NotImplementedError
            elif self.method == 'containment':
                q_result = set()
                for k, v in self.top_explanations.items():
                    if self._containment(v[0], s) > self.threshold:
                        q_result.add(k)

            if len(q_result) == 0:
                if len(self.top_explanations) <= self.k - 1:
                    self._update_min(explanation, score)
                    self.top_explanations[explanation] = (s, score)
                    return 0
        return -1


def get_subset(explanation):
    subset = X_train_orig.copy()
    for predicate in explanation:
        attr = predicate.split("=")[0].strip(' ')
        val = int(predicate.split("=")[1].strip(' '))
        subset = subset[subset[attr] == val]
    return subset.index


def first_order_group_influence(U, del_L_del_theta):
    n = len(X_train)
    return 1 / n * np.sum(np.dot(del_L_del_theta[U, :], hinv), axis=0)


def second_order_group_influence(U, del_L_del_theta):
    u = len(U)
    s = len(X_train)
    p = u / s
    c1 = (1 - 2 * p) / (s * (1 - p) ** 2)
    c2 = 1 / ((s * (1 - p)) ** 2)
    del_L_del_theta_sum = np.sum(del_L_del_theta[U, :], axis=0)
    hinv_del_L_del_theta = np.matmul(hinv, del_L_del_theta_sum)
    hessian_U_hinv_del_L_del_theta = np.sum(np.matmul(hessian_all_points[U, :], hinv_del_L_del_theta), axis=0)
    term1 = c1 * hinv_del_L_del_theta
    term2 = c2 * np.matmul(hinv, hessian_U_hinv_del_L_del_theta)
    sum_term = (term1 + term2 * len(X_train))
    return sum_term


def fairness_specific_precompute(new):
    global metric, v1, hinv, hinv_v, model, candidates, metric_val
    metric = ['statistical parity', 'equal opportunity', 'predictive parity'].index(new)
    metric_val = metric_vals[metric]
    if metric == 0:
        v1 = del_spd_del_theta(model, X_test_orig, X_test, dataset.value)
    elif metric == 1:
        v1 = del_tpr_parity_del_theta(model, X_test_orig, X_test, y_test, dataset.value)
    elif metric == 2:
        v1 = del_predictive_parity_del_theta(model, X_test_orig, X_test, y_test, dataset.value)
    hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
    hinv_v = s_test(model, X_train, y_train, v1, hinv=hinv)

    model.fit(X_train, y_train)

    attributes = []
    attributeValues = []
    first_order_influences = []
    second_order_influences = []
    fractionRows = []

    for col in X_train_orig.columns:
        if dataset.value == 'german':
            if "purpose" in col or "housing" in col:  # dummy variables purpose=0 doesn't make sense
                vals = [1]
            else:
                vals = X_train_orig[col].unique()
        elif dataset.value == 'adult':
            vals = X_train_orig[col].unique()
        elif dataset.value == 'sqf':
            vals = X_train_orig[col].unique()
        else:
            raise NotImplementedError
        for val in vals:
            idx = X_train_orig[X_train_orig[col] == val].index
            if len(idx) / len(X_train) > float(sup_lb.value) / 100:
                X = np.delete(X_train, idx, 0)
                y = y_train.drop(index=idx, inplace=False)
                if len(y.unique()) > 1:
                    idx = X_train_orig[X_train_orig[col] == val].index

                    # First-order subset influence
                    params_f_1 = first_order_group_influence(idx, del_L_del_theta)
                    del_f_1 = np.dot(v1.transpose(), params_f_1)

                    # Second-order subset influence
                    params_f_2 = second_order_group_influence(idx, del_L_del_theta)
                    del_f_2 = np.dot(v1.transpose(), params_f_2)

                    attributes.append(col)
                    attributeValues.append(val)
                    first_order_influences.append(del_f_1)
                    second_order_influences.append(del_f_2)
                    fractionRows.append(len(idx) / len(X_train) * 100)
    expl = [attributes, attributeValues, first_order_influences, second_order_influences, fractionRows]
    expl = np.array(expl).T.tolist()

    explanations = pd.DataFrame(expl, columns=["attributes", "attributeValues", "first_order_influences",
                                               "second_order_influences", "fractionRows"])
    explanations['second_order_influences'] = explanations['second_order_influences'].astype(float)
    explanations['first_order_influences'] = explanations['first_order_influences'].astype(float)
    explanations['fractionRows'] = explanations['fractionRows'].astype(float)
    candidates = copy.deepcopy(explanations)
    candidates.loc[:, 'score'] = candidates.loc[:, 'second_order_influences'] * 100 / candidates.loc[:, 'fractionRows']

    removal_explain.disabled = False


def removal_based_explanation():
    global explanations
    candidates_all = []
    total_rows = len(X_train_orig)
    support_small = 0.3
    support = float(sup_lb.value) / 100
    del_f_threshold = 0.1 * metric_val

    # Generating 1-candidates
    candidates_1 = []
    for i in range(len(candidates)):
        candidate_i = candidates.iloc[i]
        if ((candidate_i["fractionRows"] >= support_small) or
                ((candidate_i["fractionRows"] >= support) & (candidate_i["second_order_influences"] > del_f_threshold))
        ):
            attr_i = candidate_i["attributes"]
            val_i = int(float(candidate_i["attributeValues"]))
            idx = X_train_orig[X_train_orig[attr_i] == val_i].index
            predicates = [attr_i + '=' + str(val_i)]
            candidate = [predicates, candidate_i["fractionRows"],
                         candidate_i["score"], candidate_i["second_order_influences"], idx]
            candidates_1.append(candidate)

    print("Generated: ", len(candidates_1), " 1-candidates")
    candidates_1.sort()

    for i in range(len(candidates_1)):
        if float(candidates_1[i][2]) >= support:  # if score > top-k, keep in candidates, not otherwise
            candidates_all.insert(len(candidates_all), candidates_1[i])

    # Generating 2-candidates
    candidates_2 = []
    for i in range(len(candidates_1)):
        predicate_i = candidates_1[i][0][0]
        attr_i = predicate_i.split("=")[0]
        val_i = int(float(predicate_i.split("=")[1]))
        sup_i = candidates_1[i][1]
        idx_i = candidates_1[i][-1]
        for j in range(i):
            predicate_j = candidates_1[j][0][0]
            attr_j = predicate_j.split("=")[0]
            val_j = int(float(predicate_j.split("=")[1]))
            sup_j = candidates_1[j][1]
            idx_j = candidates_1[j][-1]
            if attr_i != attr_j:
                idx = idx_i.intersection(idx_j)
                fractionRows = len(idx) / total_rows * 100
                isCompact = True
                # pattern is not compact if intersection equals one of its parents
                if fractionRows == min(sup_i, sup_j):
                    isCompact = False
                if fractionRows / 100 >= support:
                    params_f_2 = second_order_group_influence(idx, del_L_del_theta)
                    del_f_2 = np.dot(v1.transpose(), params_f_2)
                    score = del_f_2 * 100 / fractionRows
                    if ((fractionRows / 100 >= support_small) or
                            ((score > candidates_1[i][2]) & (score > candidates_1[j][2]))):
                        predicates = [attr_i + '=' + str(val_i), attr_j + '=' + str(val_j)]
                        candidate = [sorted(predicates, key=itemgetter(0)), len(idx) * 100 / total_rows,
                                     score, del_f_2, idx]
                        candidates_2.append(candidate)
                        if isCompact:
                            candidates_all.append(candidate)
    print("Generated: ", len(candidates_2), " 2-candidates")

    # Recursively generating the rest
    candidates_L_1 = copy.deepcopy(candidates_2)
    set_L_1 = set()
    iteration = 2
    while (len(candidates_L_1) > 0) & (iteration < int(lvl.value)):
        print("Generated: ", iteration)
        candidates_L = []
        for i in range(len(candidates_L_1)):
            candidate_i = set(candidates_L_1[i][0])
            sup_i = candidates_L_1[i][1]
            idx_i = candidates_L_1[i][-1]
            for j in range(i):
                candidate_j = set(candidates_L_1[j][0])
                sup_j = candidates_L_1[j][1]
                idx_j = candidates_L_1[j][-1]
                merged_candidate = sorted(candidate_i.union(candidate_j), key=itemgetter(0))
                if json.dumps(merged_candidate) in set_L_1:
                    continue
                if len(merged_candidate) == iteration + 1:
                    intersect_candidates = candidate_i.intersection(candidate_j)
                    setminus_i = list(candidate_i - intersect_candidates)[0].split("=")
                    setminus_j = list(candidate_j - intersect_candidates)[0].split("=")
                    attr_i = setminus_i[0]
                    val_i = int(setminus_i[1])
                    attr_j = setminus_j[0]
                    val_j = int(setminus_j[1])
                    if attr_i != attr_j:
                        # merge to get L list
                        idx = idx_i.intersection(idx_j)
                        fractionRows = len(idx) / len(X_train) * 100
                        isCompact = True
                        # pattern is not compact if intersection equals one of its parents
                        if fractionRows == min(sup_i, sup_j):
                            isCompact = False
                        if fractionRows / 100 >= support:
                            params_f_2 = second_order_group_influence(idx, del_L_del_theta)
                            del_f_2 = np.dot(v1.transpose(), params_f_2)

                            score = del_f_2 * 100 / fractionRows
                            if (((score > candidates_L_1[i][2]) & (score > candidates_L_1[j][2])) or
                                    (fractionRows >= support_small)):
                                candidate = [merged_candidate, fractionRows,
                                             del_f_2 * len(X_train) / len(idx), del_f_2, idx]
                                candidates_L.append(candidate)
                                set_L_1.add(json.dumps(merged_candidate))
                                if isCompact:
                                    candidates_all.insert(len(candidates_all), candidate)
        set_L_1 = set()
        print("Generated:", len(candidates_L), " ", str(iteration + 1), "-candidates")
        candidates_L_1 = copy.deepcopy(candidates_L)
        candidates_L_1.sort()
        iteration += 1

    candidates_support_3_compact = copy.deepcopy(candidates_all)
    print(len(candidates_support_3_compact))
    candidates_df_3_compact = pd.DataFrame(candidates_support_3_compact,
                                           columns=["predicates", "support", "score", "2nd-inf", 'idx'])

    candidates_df_3_compact = candidates_df_3_compact[
        candidates_df_3_compact['support'] < float(sup_ub.value)].sort_values(by=['score'], ascending=False)
    containment_df = candidates_df_3_compact.sort_values(by=['score'], ascending=False).copy()

    topk = Topk(method='containment', threshold=float(containment_th.value) / 100, k=5)
    for row_idx in range(len(containment_df)):
        row = containment_df.iloc[row_idx]
        explanation, score = row[0], row[2]
        topk.update(explanation, score)
        if len(topk.top_explanations) == topk.k:
            break

    explanations = list(topk.top_explanations.keys())
    metric_idx = ['statistical parity', 'equal opportunity', 'predictive parity'].index(metric_sel.value)
    supports = list()
    scores = list()
    gt_scores = list()
    infs = list()
    gts = list()
    new_accs = list()
    for e in explanations:
        idx = get_subset(json.loads(e))
        X = np.delete(X_train, idx, 0)
        y = y_train.drop(index=idx, inplace=False)
        model.fit(np.array(X), np.array(y))
        y_pred = model.predict_proba(np.array(X_test))
        new_acc = computeAccuracy(y_test, y_pred)
        inf_gt = computeFairness(y_pred, X_test_orig, y_test, metric_idx, dataset.value) - metric_val

        condition = candidates_df_3_compact.predicates.apply(lambda x: x == json.loads(e))
        supports.append(float(candidates_df_3_compact[condition]['support']))
        scores.append(float(candidates_df_3_compact[condition]['score']))
        infs.append(float(candidates_df_3_compact[condition]['2nd-inf']))
        gts.append(inf_gt / (-metric_val))
        gt_scores.append(inf_gt * 100 / float(candidates_df_3_compact[condition]['support']))
        new_accs.append(new_acc)

    expl = [explanations, supports, scores, gt_scores, infs, gts, new_accs]
    expl = np.array(expl).T.tolist()

    explanations = pd.DataFrame(expl, columns=["explanations", "support", "score", "gt-score",
                                               "2nd-inf(%)", "gt-inf(%)", "new-acc"])
    explanations['score'] = explanations['score'].astype(float)
    explanations['gt-score'] = explanations['gt-score'].astype(float)
    explanations['support'] = explanations['support'].astype(float)
    explanations['2nd-inf(%)'] = explanations['2nd-inf(%)'].astype(float) / (-metric_val)
    explanations['gt-inf(%)'] = explanations['gt-inf(%)'].astype(float)
    explanations['new-acc'] = explanations['new-acc'].astype(float)

    pd.set_option('max_colwidth', 100)
    explanations.sort_values(by=['score'], ascending=False)
    top_explanations = explanations.copy().reset_index(drop=True)

    print(top_explanations)
    source_rmv.data = dict()
    source_rmv.data['explanations'] = top_explanations["explanations"].apply(proc_expls, dataset.value)
    source_rmv.data['support'] = top_explanations["support"]
    source_rmv.data['score'] = top_explanations["score"]
    source_rmv.data['second_infs'] = [value * 100 for value in top_explanations["2nd-inf(%)"]]

    source_fot.data = dict()
    source_fot.data['explanations'] = ["install_rate: > 2 % ∧ age: ≥ 45 ∧ telephone: registered",
                                       "install_rate: > 2 % ∧ age: ≥ 45 ∧ telephone: not registered",
                                       "install_rate: > 2 % ∧ age: ≤ 45 ∧ status of checking account: < 0 DM"]
    source_fot.data['support'] = ["8.00", "9.38", "7.08"]
    source_fot.data['score'] = ["7.08", "0.55", "1.03"]
    source_fot.data['second_infs'] = ["56.64", "5.17", "18.70"]


def proc_expls(expl_line, df_name="german"):
    if df_name == "german":
        status_d = {v: k for k, v in {'< 0DM': 0, '0 ~ 200DM': 1, '≥ 200DM / salary assignment for at least 1 yaer': 2,
                                      'no checking account': 3}.items()}
        credit_hist_d = {v: k for k, v in {'critical account/ other credits existing (not at this bank)': 0,
                                           'delay in paying off in the past': 1,
                                           'existing credits paid back duly till now': 2,
                                           'all credits at this bank paid back duly': 3,
                                           'no credits taken/ all credits paid back duly': 4}.items()}
        savings_d = {v: k for k, v in {'< 100DM': 0, '100DM ~ 500DM': 1, '500DM ~ 1000DM': 2, '≥ 1000DM': 3,
                                       'unknown/ no savings account': 4}.items()}
        employment_d = {v: k for k, v in {'unemployed': 0, '< 1 year': 1, '1 year ~ 4 years': 2, '4 years ~ 7 years': 3,
                                          '≥ 7 years': 4}.items()}
        gender_d = {v: k for k, v in {'male': 1, 'female': 0}.items()}
        debtors_d = {v: k for k, v in {'none': 0, 'co-applicant': 1, 'guarantor': 2}.items()}
        property_d = {v: k for k, v in
                      {'real estate': 3, 'building society savings agreement/ life insurance': 2, 'car or other': 1,
                       'unknown / no property': 0}.items()}
        job_d = {v: k for k, v in {'unemployed/ unskilled - non-resident': 0, 'unskilled - resident': 1,
                                   'skilled employee / official': 2,
                                   'management/ self-employed/ highly qualified employee/ officer': 3}.items()}
        install_plans_d = {v: k for k, v in {'bank': 1, 'stores': 1, 'none': 0}.items()}
        telephone_d = {v: k for k, v in {'none': 0, 'yes, registered under the customers name': 1}.items()}
        foreign_worker_d = {v: k for k, v in {'yes': 1, 'no': 0}.items()}
        age_d = {1: '≥ 45', 0: '< 45'}
        credit_amt_d = {0: '≤ 2000', 1: '2000 ~ 5000', 2: ' > 5000'}
        duration_d = {0: '≤ 12', 1: '12 ~ 24', 2: '24 ~ 36', 3: '> 36'}

    elif df_name == "adult":
        income_d = {v: k for k, v in {'≤ 50K': 0, '> 50K': 1}.items()}
        age_d = {v: k for k, v in {'< 45': 0, '≥ 45': 1}.items()}
        workclass_d = {v: k for k, v in
                       {'Never-worked': 0, 'Without-pay': 1, 'State-gov': 2, 'Local-gov': 3, 'Federal-gov': 4,
                        'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'Private': 7}.items()}
        education_d = {v: k for k, v in
                       {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6,
                        '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Bachelors': 10, 'Prof-school': 11,
                        'Assoc-acdm': 12, 'Assoc-voc': 13, 'Masters': 14, 'Doctorate': 15}.items()}
        marital_d = {v: k for k, v in
                     {'Married-civ-spouse': 2, 'Divorced': 1, 'Never-married': 0, 'Separated': 1, 'Widowed': 1,
                      'Married-spouse-absent': 2, 'Married-AF-spouse': 2}.items()}
        relationship_d = {v: k for k, v in
                          {'Wife': 1, 'Own-child': 0, 'Husband': 1, 'Not-in-family': 0, 'Other-relative': 0,
                           'Unmarried': 0}.items()}
        race_d = {v: k for k, v in
                  {'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0, 'Other': 0, 'Black': 0}.items()}
        gender_d = {v: k for k, v in {'Male': 1, 'Female': 0}.items()}
        hours_d = {v: k for k, v in {'> 40': 1, '≤ 40': 0}.items()}

    out = ""
    expl_lst = eval(expl_line)
    for expl in expl_lst:
        expl_type = expl.split("=")[0]
        expl_value = int(expl.split("=")[-1])
        d = expl_type + "_d"
        try:
            expl_d = eval(d)
        except NameError:
            out += expl_type + ": " + expl + " ∧ "
            continue

        meaning = expl_d[expl_value]
        out += expl_type + ": " + meaning + " ∧ "

    out = out[:-3]
    return out


update_dataset_preview()

controls = [dataset, clf, train, pre_compute]
col11 = column(*controls, name="tab1_inp")

metrics = [acc, spd, tpr, ppr, pre_compute_percent]
col12 = column(*metrics, sizing_mode='stretch_width')

train.on_click(handler=update_pre)
pre_compute.on_click(handler=pre_computation)
dataset.on_change('value', lambda attr, old, new: update_dataset_preview())
tab1_inp = col11
tab1_table = table
curdoc().add_root(tab1_inp)
curdoc().add_root(tab1_table)
# section1 = row(col11, table, col12, sizing_mode='stretch_width')
# tab1 = Panel(child=section1, title="Preparation")

settings = [lvl, sup_lb, sup_ub, containment_th, metric_sel, removal_explain]
removal_explain.on_click(handler=removal_based_explanation)
metric_sel.on_change('value', lambda attr, old, new: fairness_specific_precompute(new))
col21 = column(*settings, name='tab2_inp')
curdoc().add_root(col21)
curdoc().add_root(table_rmv)
curdoc().add_root(table_fot)
curdoc().add_root(table_upd)
# section2 = row(col21, table_rmv, sizing_mode='stretch_width')
# tab2 = Panel(child=section2, title="Removal-based Explanation")

row30 = row(add_attr, remove_attr, sizing_mode='stretch_width')
row31 = row(update_attr_1, update_val_1, updated_val_1, sizing_mode='stretch_width')
update_attr_1.on_change('value', lambda attr, old, new: update_val1_option())
row32 = row(update_attr_2, update_val_2, updated_val_2, sizing_mode='stretch_width')
update_attr_2.on_change('value', lambda attr, old, new: update_val2_option())
row33 = row(update_attr_3, update_val_3, updated_val_3, sizing_mode='stretch_width')
update_attr_3.on_change('value', lambda attr, old, new: update_val3_option())
row34 = row(update_attr_4, update_val_4, updated_val_4, sizing_mode='stretch_width')
update_attr_4.on_change('value', lambda attr, old, new: update_val4_option())
# update_data_view.on_click(handler=update_dataset_upd_preview)

add_attr.on_click(handler=add_attr_handler)
remove_attr.on_click(handler=remove_attr_handler)
update_explain.on_click(handler=update_comparison_fig)
col31 = column(row30, row31, row32, row33, row34, update_explain, name='tab3_inp', sizing_mode='stretch_width')
curdoc().add_root(col31)

curdoc().title = "Gopher Demo"
# curdoc().theme = 'night_sky'

# update_pre()
pre_compute.disabled = True
removal_explain.disabled = True
# tab2.disabled = True
