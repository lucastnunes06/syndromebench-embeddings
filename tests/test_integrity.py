"""Additional checks for packaging integrity and group-aware preprocessing."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import data_processing as d
import eda
from knn_classification import check_subject_syndrome_consistency, _safe_group_splits


def test_reject_malformed_embedding_vectors():
    valid=np.ones(320)
    data={'x':{'person1':{'good':valid,'bad_dims':[1,2],'nan':np.full(320,np.nan)}}}
    frame=d.flatten_data(data)
    assert len(frame)==1
    assert frame.iloc[0]['image_id']=='good'


def test_reject_no_valid_vectors():
    with pytest.raises(ValueError,match='No valid records'):
        d.flatten_data({'x':{'p':{'bad':[1,2]}}})


def test_empty_subject_dup_deduplication():
    vector=np.zeros(320)
    data={'one':{'same':{'photo':vector}},'two':{'same':{'photo':vector}}}
    frame=d.flatten_data(data)
    assert len(frame)==1


def test_subject_with_two_syndromes_must_fail():
    with pytest.raises(ValueError,match='multiple syndromes'):
        check_subject_syndrome_consistency(np.array(['s1','s2']),np.array(['subject','subject']))


def test_outer_splits_do_not_mix_subjects():
    groups=np.array([f's{j}' for j in range(30) for _ in range(2)])
    y=np.array([f'class{j%3}' for j in range(30) for _ in range(2)])
    X=np.ones((len(y),320))
    splits=_safe_group_splits(X,y,groups,n_splits=3,random_state=12)
    for train,test in splits:
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_npz_loader_does_not_require_pickle():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'features.npz'
        np.savez_compressed(p,X=np.zeros((2,320)),y=np.array(['a','b']),
                            subject_id=np.array(['s0','s1']),image_id=np.array(['i0','i1']))
        X,y,subjects,ids=eda.load_embeddings(p)
        assert X.shape==(2,320)
        assert len(subjects)==2


def test_real_serializer_roundtrip_without_pickle(tmp_path):
    source = {'class1': {'person1': {'img1': np.ones(320)}}}
    frame = d.flatten_data(source)
    d.save_flattened(frame, tmp_path)
    with np.load(tmp_path / 'embeddings.npz', allow_pickle=False) as data:
        assert data['X'].shape == (1, 320)
        assert data['y'][0] == 'class1'
        assert data['subject_id'][0] == 'person1'
