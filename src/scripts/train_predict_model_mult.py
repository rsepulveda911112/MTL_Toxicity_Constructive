import argparse
from common.load_data import load_data_mul, load_data
import os
from common.score import cal_score
from model.model_mul import SequenceModel
from sklearn.model_selection import train_test_split


def main(parser):
    args = parser.parse_args()
    training_set = args.training_set
    test_set = args.test_set
    eval_set = args.eval_set
    use_cuda = args.use_cuda
    is_evaluate = args.is_evaluate
    model_dir = args.model_dir
    model_type = args.model_type
    model_name = args.model_name
    wandb_project = args.wandb_project
    is_sweeping = args.is_sweeping
    best_result_config = args.best_result_config
    exec_model(model_type, model_name, model_dir, training_set, test_set, eval_set, is_evaluate, use_cuda,
               wandb_project, is_sweeping, best_result_config)


def exec_model(model_type, model_name, model_dir, training_set, test_set, eval_set, is_evaluate, use_cuda,
               wandb_project, is_sweeping, best_result_config):
    if model_dir == '':
        df_train, _ = load_data_mul(os.getcwd() + training_set, False, ['CONST', 'TOXICIDAD'])
        #df_train = df_train[0:100]
        model = SequenceModel(model_type, model_name, use_cuda, None, [2,4], wandb_project,
                            is_sweeping, is_evaluate, best_result_config, True, len(df_train['features'][0]))
        df_eval = None
        if is_evaluate:
            if eval_set != '':
                df_eval, _ = load_data_mul(os.getcwd() + eval_set, False, ['CONST', 'TOXICIDAD'])
            else:
                df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)
        # df_train = pd.concat([df_train, df_eval])
        model.train(df_train, df_eval, ['CONST', 'TOXICIDAD'])

    else:
        model = SequenceModel(model_type, os.getcwd() + model_dir, use_cuda)

    for index, label in enumerate(['CONST', 'TOXICIDAD']):
        ###### Predict test set ########
        df_test, _ = load_data_mul(os.getcwd() + test_set, False, [label])
        df_test.rename(columns={label: 'labels'}, inplace=True)
        #df_test = df_test[0:10]
        y_predict, _ = model.predict(df_test, task_id=index)
        cal_score(df_test, y_predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_set",
                        default="/data/const_toxico_train.tsv",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--eval_set",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of eval set.")

    parser.add_argument("--test_set",
                        default="/data/test_v1.tsv",
                        type=str,
                        help="This parameter is the relative dir of eval set.")

    parser.add_argument("--use_cuda",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--is_evaluate",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if you want to split train in train and dev.")

    parser.add_argument("--model_dir",
                        default='',
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--model_type",
                        default="roberta",
                        type=str,
                        help="This parameter is the relative type of model to trian and predict.")

    parser.add_argument("--model_name",
                        default="PlanTL-GOB-ES/roberta-base-bne",
                        type=str,
                        help="This parameter is the relative name of model to trian and predict.")

    parser.add_argument("--wandb_project",
                        default="MTL_toxicity_const",
                        type=str,
                        help="This parameter is the name of wandb project.")

    parser.add_argument("--is_sweeping",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you use sweep search.")

    parser.add_argument("--best_result_config",
                        default="",
                        type=str,
                        help="This parameter is the file with best hyperparameters configuration.")

    main(parser)
