import pandas as pd
from gensim.models import KeyedVectors
from node2vec import Node2Vec
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def create_seq_prob_edge_list(sequence_details_df, stu_prob_edge_list):
    sequence_df = sequence_details_df[['sequence_id', 'sequence_problem_ids']]

    sequence_df['sequence_problem_ids'] = sequence_df['sequence_problem_ids'].apply(lambda x: x.strip('][').split(','))

    sequence_df = sequence_df.explode('sequence_problem_ids').reset_index(drop=True)

    sequence_df['sequence_problem_ids'] = 'prob_' + sequence_df['sequence_problem_ids'].astype(str)

    merged = pd.merge(left=stu_prob_edge_list, right=sequence_df, left_on='node_2', right_on='sequence_problem_ids',
                      how='outer')

    merged = merged.dropna(subset=['node_2'])

    edge_list = merged[['sequence_id', 'node_2']]

    edge_list = edge_list.drop_duplicates()

    edge_list = edge_list.dropna(subset=['sequence_id'])

    edge_list['sequence_id'] = 'seq_' + edge_list['sequence_id'].astype(str)

    edge_list = edge_list.reset_index(drop=True)

    edge_list.rename({'sequence_id': 'node_1'}, axis=1, inplace=True)

    edge_list.to_csv('./saved_files/seq_prob_edge_list.csv', index=False)

    print(f'# sequences: {len(pd.unique(edge_list.node_1))}')
    return edge_list


def create_teacher_class_edge_list(assignment_entities):
    teacher_class_test = assignment_entities[['teacher_id_x', 'class_id_x']].drop_duplicates()

    teacher_class_in_unit = assignment_entities[['teacher_id_y', 'class_id_y']].drop_duplicates()

    teacher_class_in_unit.rename({'teacher_id_y': 'node_1', 'class_id_y': 'node_2'}, axis=1, inplace=True)

    teacher_class_test.rename({'teacher_id_x': 'node_1', 'class_id_x': 'node_2'}, axis=1, inplace=True)

    teacher_class = pd.concat([teacher_class_test, teacher_class_in_unit], axis=0).drop_duplicates()

    teacher_class['node_1'] = 'te_' + teacher_class['node_1']
    teacher_class['node_2'] = 'cl_' + teacher_class['node_2']

    print(f'# teachers: {len(pd.unique(teacher_class.node_1))}')
    print(f'# classes: {len(pd.unique(teacher_class.node_2))}')
    return teacher_class


def create_student_class_edge_list(assignment_entities):
    student_class_test = assignment_entities[['student_id_x', 'class_id_x']].drop_duplicates()

    student_class_in_unit = assignment_entities[['student_id_y', 'class_id_y']].drop_duplicates()

    student_class_in_unit.rename({'student_id_y': 'node_1', 'class_id_y': 'node_2'}, axis=1, inplace=True)
    student_class_test.rename({'student_id_x': 'node_1', 'class_id_x': 'node_2'}, axis=1, inplace=True)

    student_class = pd.concat([student_class_test, student_class_in_unit], axis=0).drop_duplicates()

    student_class['node_1'] = 'stu_' + student_class['node_1']
    student_class['node_2'] = 'cl_' + student_class['node_2']

    return student_class


def create_stu_prob_edge_list(predict_assignment_rel_df, action_log_df):
    unit_test_problems = predict_assignment_rel_df[['unit_test_problem_id', 'student_id']].drop_duplicates()

    print(len(pd.unique(unit_test_problems.student_id)))
    predict_assignment_rel_df = predict_assignment_rel_df[
        ['student_id', 'in_unit_assignment_log_id']].drop_duplicates()

    predict_assignment_rel_df = pd.merge(left=predict_assignment_rel_df,
                                         right=action_log_df[['assignment_log_id', 'problem_id']].dropna(),
                                         left_on='in_unit_assignment_log_id', right_on='assignment_log_id')

    predict_assignment_rel_df = predict_assignment_rel_df.drop(['in_unit_assignment_log_id', 'assignment_log_id'],
                                                               axis=1).drop_duplicates()

    in_unit_problems = predict_assignment_rel_df

    unit_test_problems['student_id'] = 'stu_' + unit_test_problems['student_id'].astype(str)
    unit_test_problems.rename({'unit_test_problem_id': 'problem_id'}, axis=1, inplace=True)
    unit_test_problems['problem_id'] = 'prob_' + unit_test_problems['problem_id'].astype(str)

    in_unit_problems['student_id'] = 'stu_' + in_unit_problems['student_id'].astype(str)
    in_unit_problems['problem_id'] = 'prob_' + in_unit_problems['problem_id'].astype(str)

    # %%
    unit_test_problems = unit_test_problems.reset_index(drop=True)

    unit_test_problems = unit_test_problems[['student_id', 'problem_id']]

    in_unit_problems = in_unit_problems.reset_index(drop=True)

    print(f'# unit test problems: {len(pd.unique(unit_test_problems.problem_id))}')
    print(f'# in-unit problems: {len(pd.unique(in_unit_problems.problem_id))}')

    edge_list = pd.concat([unit_test_problems, in_unit_problems], axis=0).drop_duplicates()
    edge_list.rename({'problem_id': 'node_2', 'student_id': 'node_1'}, axis=1, inplace=True)
    edge_list = edge_list[['node_1', 'node_2']]
    edge_list.to_csv('./saved_files/stu_prob_edge_list.csv', index=False)

    print(f'# students: {len(pd.unique(edge_list.node_1))}')
    print(f'# total problems: {len(pd.unique(edge_list.node_2))}')
    return edge_list


def create_embeddings(edge_list, embedding_filename, embedding_model_filename, dimension=64):
    try:
        embedding_model = KeyedVectors.load_word2vec_format(embedding_filename)

    except (OSError, IOError) as e:
        graph = nx.from_pandas_edgelist(edge_list, "node_1", "node_2")

        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(graph, dimensions=dimension, walk_length=10, num_walks=100,
                            workers=64)  # Use temp_folder for big graphs

        # Embed nodes
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Save embeddings for later use
        model.wv.save_word2vec_format(embedding_filename)

        # Save model for later use
        model.save(embedding_model_filename)
        embedding_model = KeyedVectors.load_word2vec_format(embedding_filename)

    return embedding_model


def combine_dataset_with_embeddings(dataset, embedding_model):
    all_embeddings = []

    for index, row in dataset.iterrows():
        # embedding = np.concatenate((embedding_model['prob_' + row['problem_id']],
        #                             # embedding_model['te_' + row['teacher_id']],
        #                             # embedding_model['cl_' + row['class_id']],
        #                             # embedding_model['seq_' + row['sequence_id']],
        #                             embedding_model['stu_' + row['student_id']]))
        embedding = embedding_model['prob_' + row['problem_id']]
        all_embeddings.append(embedding)

    embedding_df = pd.DataFrame(all_embeddings)

    dataset = dataset.drop(
        ['problem_id', 'student_id', 'teacher_id', 'class_id', 'sequence_id', 'unit_test_assignment_log_id'],
        axis=1).reset_index(drop=True)

    dataset = pd.concat([dataset, embedding_df], axis=1)

    dataset.columns = dataset.columns.astype(str)

    return dataset


def create_embedding_dataset(dataset, embedding_model):
    all_embeddings = []

    for index, row in dataset.iterrows():
        embedding = np.concatenate((embedding_model['prob_' + row['problem_id']],
                                    # embedding_model['te_' + row['teacher_id']],
                                    # embedding_model['cl_' + row['class_id']],
                                    # embedding_model['seq_' + row['sequence_id']],
                                    embedding_model['stu_' + row['student_id']]))
        all_embeddings.append(embedding)

    embedding_df = pd.DataFrame(all_embeddings)

    dataset = pd.concat([dataset['score'], embedding_df], axis=1)

    dataset.columns = dataset.columns.astype(str)

    return dataset


def evaluate_model(y_test, test_predictions, test_probs):
    test_auc = roc_auc_score(y_test, test_probs)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1_score = f1_score(y_test, test_predictions)

    print(f'Evaluation Results')
    print(f'ROC_AUC: {test_auc}')
    print(f'Accuracy: {test_accuracy}')
    print(f'Precision: {test_precision}')
    print(f'Recall: {test_recall}')
    print(f'F1-Score: {test_f1_score}')
