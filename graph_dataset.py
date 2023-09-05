from utils import *
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--setting', default=3, required=True, type=int, help='dataset setting')

args = parser.parse_args()

if (args.setting > 3) or (args.setting < 2):
    sys.exit('Only setting 2(II) and 3(III) are implemented in this code.')

train_df = pd.read_csv('./data/training_unit_test_scores.csv')

evaluation_df = pd.read_csv('./data/evaluation_unit_test_scores.csv')

assignment_df = pd.read_csv('./data/assignment_details.csv')

assignment_rel_df = pd.read_csv('./data/assignment_relationships.csv')

action_log_df = pd.read_csv('./data/action_logs.csv')

sequence_df = pd.read_csv('./data/sequence_details.csv')

final_eval_dataset = pd.read_pickle('./saved_files/evaluation_data_with_sptcs.pkl')

final_train_dataset = pd.read_pickle('./saved_files/train_data_with_sptcs.pkl')

unit_test_scores = pd.read_csv('./data/unit_test_scores.csv')

predict_assignments = pd.concat([train_df, evaluation_df], axis=0).reset_index(drop=True)
predict_assignments.rename({'problem_id': 'unit_test_problem_id'}, axis=1, inplace=True)
predict_assignments.drop(['id', 'score'], axis=1, inplace=True)
predict_assignment_rel_df = pd.merge(left=predict_assignments, right=assignment_rel_df,
                                     left_on='assignment_log_id',
                                     right_on='unit_test_assignment_log_id').drop('assignment_log_id', axis=1)

predict_assignment_rel_df = pd.merge(left=predict_assignment_rel_df,
                                     right=assignment_df[['assignment_log_id', 'teacher_id', 'class_id', 'student_id']],
                                     left_on='unit_test_assignment_log_id',
                                     right_on='assignment_log_id').drop('assignment_log_id', axis=1)

stu_prob_edges = create_stu_prob_edge_list(predict_assignment_rel_df, action_log_df)

seq_prob_edges = create_seq_prob_edge_list(sequence_df, stu_prob_edges)

stu_prob_seq_edges = pd.concat([stu_prob_edges, seq_prob_edges], axis=0).reset_index(drop=True)

assignment_entities = pd.merge(predict_assignment_rel_df,
                               assignment_df[['assignment_log_id', 'teacher_id', 'class_id', 'student_id']],
                               left_on='in_unit_assignment_log_id', right_on='assignment_log_id').drop(
    'assignment_log_id', axis=1)

teacher_class_edges = create_teacher_class_edge_list(assignment_entities)

student_class_edges = create_student_class_edge_list(assignment_entities)

edge_list = pd.concat([stu_prob_seq_edges, teacher_class_edges], axis=0)

edge_list = pd.concat([edge_list, student_class_edges], axis=0)

edge_list.to_csv('./saved_files/all_nodes_edge_list.csv', index=False)

print(f'# edges: {len(edge_list)}')

STU_PROB_EMBEDDING_FILENAME = './models/all_nodes_node2vec_embeddings.emb'
STU_PROB_EMBEDDING_MODEL_FILENAME = './models/all_nodes_node2vec_embeddings.model'
stu_prob_embedding_model = create_embeddings(edge_list=edge_list,
                                             embedding_filename=STU_PROB_EMBEDDING_FILENAME,
                                             embedding_model_filename=STU_PROB_EMBEDDING_MODEL_FILENAME,
                                             dimension=32)

final_eval_dataset['score'] = pd.merge(final_eval_dataset, unit_test_scores,
                                       left_on=['unit_test_assignment_log_id', 'problem_id'],
                                       right_on=['assignment_log_id', 'problem_id'], how='inner')['score_y']

if args.setting == 3:
    final_eval_dataset = combine_dataset_with_embeddings(final_eval_dataset, stu_prob_embedding_model)
    final_train_dataset = combine_dataset_with_embeddings(final_train_dataset, stu_prob_embedding_model)

    final_eval_dataset = final_eval_dataset.drop(columns=['id'])
else:
    final_eval_dataset = create_embedding_dataset(final_eval_dataset, stu_prob_embedding_model)
    final_train_dataset = create_embedding_dataset(final_train_dataset, stu_prob_embedding_model)


final_eval_dataset.to_pickle(f'./saved_files/eval_setting_{args.setting}.pkl')
final_eval_dataset.to_csv(f'./saved_files/eval_setting_{args.setting}.csv', index=False)

final_train_dataset.to_pickle(f'./saved_files/train_setting_{args.setting}.pkl')
final_train_dataset.to_csv(f'./saved_files/train_setting_{args.setting}.csv', index=False)

print(f'# columns in training set: {len(final_train_dataset.columns)}')
print(f'# columns in evaluation set: {len(final_eval_dataset.columns)}')
