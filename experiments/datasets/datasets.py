import os, sys
sys.path.append(os.getcwd())
import pickle as pkl
from urllib import request
import pandas as pd
import numpy as np
from zipfile import ZipFile

dataset_list = []


def load_datasets(datasets=None):
    """
    Args:
        datasets (list of str): Lists of datasets to load by name. If None, all available datasets are loaded iteratively.
    """
    if datasets is None:
        datasets = [d.name for d in dataset_list]
    for dataset in dataset_list:
        if dataset.name in datasets:
            yield dataset.load()


def classproperty(method):
    class ClassPropertyDescriptor:
        def __init__(self, method):
            self.method = method
        def __get__(self, obj, objtype=None):
            return self.method(objtype)
    return ClassPropertyDescriptor(method)


class Dataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.n_examples, self.n_features = self.data.shape
        self.n_classes = len(set(self.target))

    @property
    def data(self):
        return self.dataframe.loc[:, self.dataframe.columns != 'class'].to_numpy(dtype=float)

    @property
    def target(self):
        return self.dataframe.loc[:, 'class'].to_numpy()

    def __repr__(self):
        return f'Dataset f{type(self).name} with {self.n_examples} examples and {self.n_features} features'

    @classproperty
    def path_to_raw_file(cls):
        return os.path.dirname(__file__) + '/raw/' + cls.name + '.raw'

    @classmethod
    def load(cls):
        if not os.path.exists(cls.path_to_raw_file):
            cls.download_dataset()

        return cls(cls.create_dataframe())

    @classmethod
    def download_dataset(cls):
        content = request.urlopen(cls.url)
        os.makedirs(os.path.dirname(__file__) + '/raw/', exist_ok=True)
        with open(cls.path_to_raw_file, 'wb') as file:
            for line in content:
                file.write(line)

    @classmethod
    def create_dataframe(cls):
        raise NotImplementedError


class BreastCancerWisconsinDiagnostic(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    name = "breast_cancer_wisconsin_diagnostic"
    bibtex_label = "street1993nuclear"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = ['id', 'class'] + [f'attr {i}' for i in range(30)]
            df = pd.read_csv(file, names=col_names, header=None)
            df.drop(columns=col_names[0], inplace=True)
        return df
dataset_list.append(BreastCancerWisconsinDiagnostic)

class Cardiotocography10(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"
    name = "cardiotocography_10"
    bibtex_label = "ayres2000sisporto"
    @classmethod
    def create_dataframe(cls):
        with pd.ExcelFile(cls.path_to_raw_file) as file:
            df = pd.read_excel(file, sheet_name=file.sheet_names[1], header=0, skiprows=[0] + [i for i in range(2128, 2131)])
            cols = list(df)
            cols_to_drop = cols[:10] + cols[31:43] + cols[-2:]
            df.drop(columns=cols_to_drop, inplace=True)
            df.rename(columns={'CLASS':'class'}, inplace=True)
        return df
dataset_list.append(Cardiotocography10)

class ClimateModelSimulationCrashes(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
    name = "climate_model_simulation_crashes"
    bibtex_label = "lucas2013failure"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0, delim_whitespace=True)
            df.drop(columns=list(df)[:2], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(ClimateModelSimulationCrashes)

class ConnectionistBenchSonar(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    name = "connectionist_bench_sonar"
    bibtex_label = "gorman1988analysis"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(ConnectionistBenchSonar)

class DiabeticRetinopathyDebrecen(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
    name = "diabetic_retinopathy_debrecen"
    bibtex_label = "antal2014ensemble"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(24)))
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(DiabeticRetinopathyDebrecen)

class Fertility(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    name = "fertility"
    bibtex_label = "gil2012predicting"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(Fertility)

class HabermansSurvival(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    name = "habermans_survival"
    bibtex_label = "haberman1976generalized"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(HabermansSurvival)

class ImageSegmentation(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
    name = "image_segmentation"
    bibtex_label = None
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, skiprows=list(range(4)))
            df.rename(columns={list(df)[0]:'class'}, inplace=True)
        return df
dataset_list.append(ImageSegmentation)

class Ionosphere(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    name = "ionosphere"
    bibtex_label = "sigillito1989classification"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(Ionosphere)

class Iris(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    name = "iris"
    bibtex_label = "fisher1936use"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        return df
dataset_list.append(Iris)

class Parkinson(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    name = "parkinson"
    bibtex_label = "little2007exploiting"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=0)
            df.rename(columns={'status':'class'}, inplace=True)
            df.drop(columns='name', inplace=True)
        return df
dataset_list.append(Parkinson)

class PlanningRelax(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt"
    name = "planning_relax"
    bibtex_label = "bhatt2012planning"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep='\t', )
            df.drop(columns=list(df)[-1], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(PlanningRelax)

class QSARBiodegradation(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
    name = "qsar_biodegradation"
    bibtex_label = "mansouri2013quantitative"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, sep=';')
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(QSARBiodegradation)

class Seeds(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    name = "seeds"
    bibtex_label = "charytanowicz2010complete"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(Seeds)

class Spambase(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    name = "spambase"
    bibtex_label = None
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(Spambase)

class VertebralColumn3C(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    name = "vertebral_column_3c"
    bibtex_label = "berthonnaud2005analysis"
    @classmethod
    def create_dataframe(cls):
        with ZipFile(cls.path_to_raw_file, 'r') as zipfile:
            with zipfile.open('column_3C.dat') as file:
                df = pd.read_csv(file, header=None, delim_whitespace=True)
                df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(VertebralColumn3C)

class WallFollowingRobot24(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data"
    name = "wall_following_robot_24"
    bibtex_label = "freire2009short"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(WallFollowingRobot24)

class Wine(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    name = "wine"
    bibtex_label = "aeberhard1994comparative"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            col_names = [
                'class',
                'Alcohol',
                'Malic acid',
                'Ash',
                'Alcalinity of ash',
                'Magnesium',
                'Total phenols',
                'Flavanoids',
                'Nonflavanoid phenols',
                'Proanthocyanins',
                'Color intensity',
                'Hue',
                'OD280/OD315 of diluted wines',
                'Proline'
            ]
            df = pd.read_csv(file, names=col_names, header=None)
        return df
dataset_list.append(Wine)

class Yeast(Dataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    name = "yeast"
    bibtex_label = "horton1996probabilistic"
    @classmethod
    def create_dataframe(cls):
        with open(cls.path_to_raw_file, 'r') as file:
            df = pd.read_csv(file, header=None, delim_whitespace=True)
            df.drop(columns=list(df)[0], inplace=True)
            df.rename(columns={list(df)[-1]:'class'}, inplace=True)
        return df
dataset_list.append(Yeast)


if __name__ == "__main__":
    for i, d in enumerate(load_datasets()):
        assert not np.isnan(d.data.sum())
        print(i, d.name, d.n_examples, d.n_classes)

