import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np

from experiments.datasets.datasets import dataset_list

from partitioning_machines import func_to_cmd


class MeanWithStd(float, p2l.TexObject):
    def __new__(cls, array):
        mean = array.mean()
        instance = super().__new__(cls, mean)
        instance.mean = mean
        instance.std = array.std()
        return instance

    def __format__(self, format_spec):
        if np.isnan(self.mean):
            return 'N/A'
        return f'${format(self.mean, format_spec)} \pm {format(self.std, format_spec)}$'


def build_table(dataset, exp_name):
    dataset = dataset.load()
    dataset_name = dataset.name
    model_names = [
        'original',
        'cart',
        'm-cart',
        'ours',
        ]
    
    if isinstance(exp_name, str):
        exp_names = [exp_name]*4
    else:
        exp_names = exp_name

    table = p2l.Table((7, 5), float_format='.3f')
    table.body.insert(0, '\\small')
    table[0,1:] = ['Original', 'CART', 'M-CART', 'Ours']
    table[0,1:].add_rule()
    table[1:,0] = ['Train acc.', 'Test acc.', 'Leaves', 'Height', 'Time $[s]$', 'Bound']

    for i, (model, exp) in enumerate(zip(model_names, exp_names)):
        tr_acc = []
        ts_acc = []
        leaves = []
        height = []
        bound = []
        time = []
        path = './experiments/results/' + dataset_name + '/' + exp + '/' + model + '.csv'
        try:
            with open(path, 'r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                for row in reader:
                    tr_acc.append(row[2])
                    ts_acc.append(row[3])
                    leaves.append(row[4])
                    height.append(row[5])
                    try:
                        time.append(row[7])
                    except:
                        time.append(np.nan)
                    if i == 3:
                        bound.append(row[6])
                    else:
                        bound.append(np.nan)

            tr_acc = np.array(tr_acc, dtype=float)
            ts_acc = np.array(ts_acc, dtype=float)
            leaves = np.array(leaves, dtype=float)
            height = np.array(height, dtype=float)
            time = np.array(time, dtype=float)
            bound = np.array(bound, dtype=float)

            table[0+1, i+1] = MeanWithStd(tr_acc)
            table[1+1, i+1] = MeanWithStd(ts_acc)
            table[2+1, i+1] = MeanWithStd(leaves)
            table[3+1, i+1] = MeanWithStd(height)
            table[4+1, i+1] = MeanWithStd(time)
            table[5+1, i+1] = MeanWithStd(bound)
        except FileNotFoundError:
            print(f"Could not find file '{path}'")

    dataset_name = dataset_name.replace('_', ' ').title()
    dataset_citation = '' if dataset.bibtex_label is None else f'\\citep{{{dataset.bibtex_label}}}'
    table.caption = f'{dataset_name} Dataset {dataset_citation} ({dataset.n_examples} examples, {dataset.n_features} features, {dataset.n_classes} classes)'
    table[2,1:].highlight_best(highlight=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=.0025, rtol=0)
    table[3:,1:].change_format('.1f')

    return table


@func_to_cmd
def process_results(exp_name='first_exp'):
    """
    Produces Tables 2 to 20 from the paper (Appendix E). Will try to call pdflatex if installed.
    
    Args:
        exp_name (str): Name of the experiment used when the experiments were run. If no experiments by that name are found, entries are set to 'nan'.
    
    Prints in the console some compiled statistics used in the paper and the tex string used to produce the tables, and will compile it if possible.
    """
    
    doc = p2l.Document(exp_name + '_results_by_dataset', '.')
    doc.add_package('natbib')

    tables = [build_table(dataset, exp_name) for dataset in dataset_list]
    
    # Other stats
    print('Some compiled statistics used in the paper:\n')
    
    times_leaves_cart = [table[3,4].data[0][0] / table[3,2].data[0][0] for table in tables]
    print('CART tree has', sum(times_leaves_cart)/len(times_leaves_cart), 'times less leaves than our model.')
    acc_gain_vs_cart = [table[2,4].data[0][0] - table[2,2].data[0][0] for table in tables]
    print('Our model has a mean accuracy gain of', sum(acc_gain_vs_cart)/len(acc_gain_vs_cart), 'over the CART algorithm.')
    time_ours_vs_cart = [table[5,2].data[0][0] / table[5,4].data[0][0] for table in tables]
    print('It took in average', sum(time_ours_vs_cart)/len(time_ours_vs_cart), 'less time to prune the tree with our model than with the CART algorithm.')

    times_leaves_mcart = [table[3,3].data[0][0] / table[3,2].data[0][0] for table in tables]
    print('CART tree has', sum(times_leaves_mcart)/len(times_leaves_mcart), 'times less leaves than the modified CART algorithm.')
    acc_gain_vs_mcart = [table[2,3].data[0][0] - table[2,2].data[0][0] for table in tables]
    print('The modified CART algorithm has a mean accuracy gain of', sum(acc_gain_vs_mcart)/len(acc_gain_vs_mcart), 'over the CART algorithm.')

    print('\n')

    doc.body.extend(tables)
    print(doc.build(save_to_disk=False))
    
    try:
        doc.build()
    except:
        pass


if __name__ == "__main__":
    process_results()