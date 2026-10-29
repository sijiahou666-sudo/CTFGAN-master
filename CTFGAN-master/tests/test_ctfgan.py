import sys
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import numpy as np
import pytest
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from models.GAN import NetG, NetD, NetC, CFBlock
from models.DAMSM import RNN_ENCODER
from lib.contours import SobelContour, contour_loss, canny_image
from lib.ctf_data import encode_text, preprocess, ContourDataset
from ctfgan import config, train_step, train, sample, load_generator

torch.set_num_threads(2)


def test_paper_shapes_and_contour_condition():
    torch.manual_seed(7)
    g, d, c = NetG(2, 8, 16), NetD(2), NetC(2, 16)
    z, e = torch.randn(2, 8), torch.randn(2, 16)
    contour = torch.rand(2, 1, 256, 256)
    fake = g(z, e, contour)
    assert fake.shape == (2, 3, 256, 256)
    assert len(g.GBlocks) == 7 and len(d.DBlocks) == 6
    assert g.resolutions == [4, 8, 16, 32, 64, 128, 256]
    assert not torch.allclose(fake, g(z, e, torch.zeros_like(contour)))
    assert c(d(fake), e, contour).shape == (2, 1, 1, 1)
    assert not torch.allclose(c(d(fake), e, contour), c(d(fake), e, 1-contour))


@pytest.mark.parametrize('tf,cf', [(True, False), (False, True), (False, False), (True, True)])
def test_ablation_backward(tf, cf):
    g = NetG(2, 8, 16, 32, use_tf=tf, use_cf=cf)
    fake = g(torch.randn(2, 8), torch.randn(2, 16), torch.rand(2, 1, 32, 32))
    loss = contour_loss(fake, torch.rand(2, 1, 32, 32))
    loss.backward()
    assert g.fc.weight.grad.abs().sum() > 0
    assert torch.isfinite(g.fc.weight.grad).all()


def test_sobel_zero_flat_gradient_and_reduction():
    sobel = SobelContour()
    assert torch.equal(sobel(torch.zeros(2, 3, 16, 16)), torch.zeros(2, 1, 16, 16))
    image = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)
    loss = contour_loss(image, target, sobel)
    assert torch.allclose(loss, contour_loss(image, target, sobel, 'mean')*256)
    loss.backward()
    assert image.grad.abs().sum() > 0 and torch.isfinite(image.grad).all()


def test_sorting_keeps_batch_one_and_pair_order():
    encoder = RNN_ENCODER(10, nhidden=16).eval()
    tokens = torch.tensor([[1, 0, 0], [2, 3, 4]])
    sentence = encode_text(encoder, tokens, torch.tensor([1, 3]))
    single = encode_text(encoder, tokens[:1], torch.tensor([1]))
    assert single.shape == (1, 16)
    assert torch.allclose(sentence[:1], single, atol=1e-6)


def test_full_train_step_with_magp():
    g, d, c = NetG(2, 8, 16, 32), NetD(2, 32), NetC(2, 16)
    og = torch.optim.Adam(g.parameters(), lr=1e-4)
    od = torch.optim.Adam(list(d.parameters())+list(c.parameters()), lr=4e-4)
    before = g.fc.weight.detach().clone()
    args = SimpleNamespace(z_dim=8, magp_weight=2., lambda_contour=.1, contour_reduction='sum')
    values = train_step(g, d, c, og, od, torch.randn(2, 3, 32, 32), torch.randn(2, 16),
                        torch.rand(2, 1, 32, 32), args, SobelContour())
    assert all(np.isfinite(v) for v in values.values())
    assert values['magp'] >= 0 and not torch.equal(before, g.fc.weight)


def fixture_data(tmp_path):
    args = config(Path(__file__).resolve().parents[1]/'code/cfg/ctfgan_coco.yml', {})
    args.data_dir, args.contour_dir, args.output_dir = map(str, [tmp_path/'data', tmp_path/'contours', tmp_path/'run'])
    args.nf, args.z_dim, args.cond_dim = 2, 8, 16
    args.TEXT.EMBEDDING_DIM = 16
    args.TEXT.CAPTIONS_PER_IMAGE = 1
    args.num_workers, args.batch_size, args.max_epoch = 0, 2, 1
    args.gen_interval = args.save_interval = 1
    root = Path(args.data_dir)
    for split, folder in [('train', 'train2014'), ('test', 'val2014')]:
        (root/split).mkdir(parents=True)
        (root/'images'/folder).mkdir(parents=True)
        with open(root/split/'filenames.pickle', 'wb') as f:
            pickle.dump(['a', 'b', 'c'], f)
        for key in ['a', 'b', 'c']:
            Image.fromarray(np.random.default_rng(ord(key)).integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(root/'images'/folder/f'{key}.jpg')
    with open(root/'captions_DAMSM.pickle', 'wb') as f:
        pickle.dump([[[1, 2]]*3, [[1, 2]]*3, {0:'<end>', 1:'a', 2:'bird'}, {'<end>':0, 'a':1, 'bird':2}], f)
    args.TEXT.DAMSM_NAME = str(root/'encoder.pth')
    torch.save(RNN_ENCODER(3, nhidden=16).state_dict(), args.TEXT.DAMSM_NAME)
    return args


def test_offline_train_checkpoint_resume_and_sample(tmp_path):
    args = fixture_data(tmp_path)
    preprocess(args)
    ds = ContourDataset(args, 'test')
    assert ds[0][0].shape == (3, 256, 256)
    assert set(ds[0][3].unique().tolist()) <= {0., 1.}
    train(args, torch.device('cpu'))
    path = Path(args.output_dir)/'latest.pth'
    g, checkpoint = load_generator(args, path, 'cpu')
    assert checkpoint['epoch'] == 1 and checkpoint['step'] == 1
    args.max_epoch = 2
    train(args, torch.device('cpu'), resume=path)
    _, checkpoint = load_generator(args, path, 'cpu')
    assert checkpoint['epoch'] == 2 and checkpoint['step'] == 2
    manifest = tmp_path/'pairs.jsonl'
    manifest.write_text(json.dumps({'text':'a bird', 'contour':str(Path(args.contour_dir)/'test/a.png')}))
    sample(args, 'cpu', path, manifest, 1, 42)
    output = Path(args.output_dir)/'samples/000000_00.png'
    assert Image.open(output).size == (256, 256)
    first = output.read_bytes()
    sample(args, 'cpu', path, manifest, 1, 42)
    assert output.read_bytes() == first


def test_config_overrides_only_explicit():
    path = Path(__file__).resolve().parents[1]/'code/cfg/ctfgan_bird.yml'
    args = config(path, {'batch_size':None, 'max_epoch':None})
    assert args.batch_size == 32 and args.max_epoch == 1220
    assert config(path, {'batch_size':4}).batch_size == 4


def test_evaluation_keeps_partial_batch_and_records_pairs(tmp_path, monkeypatch):
    from ctfgan import checkpoint as make_checkpoint, evaluate
    from models import inception
    args = fixture_data(tmp_path)
    preprocess(args)
    g = NetG(args.nf, args.z_dim, args.cond_dim)
    d, c = NetD(args.nf), NetC(args.nf, args.cond_dim)
    og = torch.optim.Adam(g.parameters())
    od = torch.optim.Adam(list(d.parameters())+list(c.parameters()))
    state = make_checkpoint(args, g, d, c, og, od, 0, 0)
    saved = tmp_path/'state.pth'
    torch.save(state, saved)
    class FakeInception(torch.nn.Module):
        BLOCK_INDEX_BY_DIM = {2048: 0}
        def __init__(self, blocks):
            super().__init__()
        def forward(self, image):
            return [image.mean((2, 3), keepdim=True)]
    monkeypatch.setattr(inception, 'InceptionV3', FakeInception)
    args.npz_path = str(tmp_path/'stats.npz')
    np.savez(args.npz_path, mu=np.zeros(3), sigma=np.eye(3))
    evaluate(args, 'cpu', saved, 1, 42)
    out = Path(args.output_dir)/'evaluation'
    metrics = json.loads((out/'metrics.json').read_text())
    records = json.loads((out/'manifest.json').read_text())
    assert metrics['samples'] == 3 and np.isfinite(metrics['FID'])
    assert [row['key'] for row in records] == ['a', 'b', 'c']
    assert [row['seed'] for row in records] == [42, 43, 44]
    assert len(list(out.glob('*.png'))) == 3
