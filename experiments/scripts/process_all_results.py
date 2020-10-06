import python2latex as p2l
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
import numpy as np

from experiments.datasets.datasets import dataset_list, load_datasets

from partitioning_machines import func_to_cmd


class MeanWithStd(float, p2l.TexObject):
    def __new__(cls, array):
        mean = array.mean()
        instance = super().__new__(cls, mean)
        instance.mean = mean
        instance.std = array.std()
        return instance

    def __format__(self, format_spec):
        return f'${format(self.mean, format_spec)} \pm {format(self.std, format_spec)}$'


@func_to_cmd
def process_results(exp_name='first_exp'):
    """
    Produces Table 1 from the paper (Appendix E). Will try to call pdflatex if installed.
    
    Args:
        exp_name (str): Name of the experiment used when the experiments were run. If no experiments by that name are found, entries are set to 'nan'.
    
    Prints in the console the tex string used to produce the tables, and will compile it if possible.
    """
    doc = p2l.Document(exp_name + '_all_results', '.')

    model_names = [
        'original',
        'cart',
        'm-cart',
        'ours',
        ]
    
    dataset_list = list(load_datasets())
    
    caption = """Mean test accuracy and standard deviation on 25 random splits of 19 datasets taken from the UCI Machine Learning Repository \\citep{Dua:2019}. In parenthesis is the total number of examples followed by the number of classes of the dataset. The best performances up to a $0.0025$ accuracy gap are highlighted in bold."""
    
    label = "results"

    alignement = r'l@{\hspace{6pt}}c@{\hspace{6pt}}c@{\hspace{6pt}}c@{\hspace{6pt}}c'
    table = doc.new(p2l.Table((len(dataset_list)+2, 5), float_format='.3f', alignment=alignement,
                              caption=caption,
                              label=label))
    table.body.insert(0, '\\small')
    
    table[0:2,0].multicell('Dataset', v_shift='-3pt')
    table[0,1:] = 'Model'
    table[1,1:] = ['Original', 'CART', 'M-CART', 'Ours']
    table[0,1:].add_rule()
    table[2:,0] = [d.name.replace('_', ' ').title() + f' ({d.n_examples}, {d.n_classes})' for d in dataset_list]
    table[1].add_rule()
    
    models_exp_name = [exp_name]*4

    for d, dataset in enumerate(dataset_list):
        for i, (model, model_exp_name) in enumerate(zip(model_names, models_exp_name)):
            ts_acc = []
            path = './experiments/results/' + dataset.name + '/' + model_exp_name + '/'
            try:
                with open(path + model + '.csv', 'r', newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    for row in reader:
                        ts_acc.append(row[3])
            except FileNotFoundError:
                ts_acc.append(np.nan)
            
            table[d+2, i+1] = MeanWithStd(np.array(ts_acc, dtype=float))

        table[d+2,1:].highlight_best(highlight=lambda content: '$\\mathbf{' + content[1:-1] + '}$', atol=0.0025, rtol=0)

    d = [dataset_list[i].load() for i in [0, 2, 3, 4, 16]]

    table[2,0] = f'BCWD\\textsuperscript{{a}} ({d[0].n_examples}, {d[0].n_classes})'
    table[4,0] = f'CMSC\\textsuperscript{{b}} ({d[1].n_examples}, {d[1].n_classes})'
    table[5,0] = f'CBS\\textsuperscript{{c}} ({d[2].n_examples}, {d[2].n_classes})'
    table[6,0] = f'DRD\\textsuperscript{{d}} ({d[3].n_examples}, {d[3].n_classes})'
    table[18,0] = f'WFR24\\textsuperscript{{e}} ({d[4].n_examples}, {d[4].n_classes})'

    table += """\n\\footnotesize \\textsuperscript{a}Breast Cancer Wisconsin Diagnostic, \\textsuperscript{b}Climate Model Simulation Crashes, \\textsuperscript{c}Connectionist Bench Sonar,\n\n\\textsuperscript{d}Diabetic Retinopathy Debrecen, \\textsuperscript{e}Wall Following Robot 24"""
    
    doc.add_package('natbib')
    
    print(doc.build(save_to_disk=False))
    
    try:
        doc.build()
    except:
        pass
    

if __name__ == "__main__":
    process_results()
