
# Participant-level stratified train/test split (no participant leakage)

X = features_df[feature_cols].copy()
y = features_df['label'].astype(int).copy()
groups = features_df['participant'].astype(str).copy()

# One row per participant is used to stratify the split.
participant_table_all = (
    features_df[['participant', 'label']]
    .drop_duplicates()
    .reset_index(drop=True)
)

assert participant_table_all.groupby('participant')['label'].nunique().max() == 1

participant_splitter = StratifiedShuffleSplit(
    n_splits=1, test_size=0.25, random_state=SEED
)

participant_train_pos, participant_test_pos = next(
    participant_splitter.split(
        participant_table_all['participant'],
        participant_table_all['label']
    )
)

train_people = set(participant_table_all.iloc[participant_train_pos]['participant'])
test_people = set(participant_table_all.iloc[participant_test_pos]['participant'])

train_idx = np.flatnonzero(groups.isin(train_people).to_numpy())
test_idx = np.flatnonzero(groups.isin(test_people).to_numpy())

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
groups_train, groups_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()
train_meta, test_meta = features_df.iloc[train_idx].copy(), features_df.iloc[test_idx].copy()

assert set(groups_train).isdisjoint(set(groups_test))
assert y_train.nunique() == 2 and y_test.nunique() == 2

print('Participant-wise stratified split successful')
print('Train recordings:', len(train_idx), '| Test recordings:', len(test_idx))
print('Train participants:', len(train_people), '| Test participants:', len(test_people))
print('Train participant class counts:')
display(participant_table_all[participant_table_all.participant.isin(train_people)].label.value_counts().rename(index={0:'HC',1:'PD'}))
print('Test participant class counts:')
display(participant_table_all[participant_table_all.participant.isin(test_people)].label.value_counts().rename(index={0:'HC',1:'PD'}))
