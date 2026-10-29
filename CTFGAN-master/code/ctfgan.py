"""Unified CTFGAN command line: preprocess, train, sample and evaluate."""
import argparse
import json
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.GAN import NetG, NetD, NetC
from models.DAMSM import RNN_ENCODER
from lib.ctf_data import ContourDataset, preprocess, encode_text
from lib.contours import SobelContour, contour_loss

ROOT = Path(__file__).resolve().parents[1]


def namespace(values):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in values.items()})


def config(path, overrides):
    values = yaml.safe_load(Path(path).read_text())
    values.update({k: v for k, v in overrides.items() if v is not None})
    args = namespace(values)
    for key in ('data_dir', 'contour_dir', 'output_dir'):
        p = Path(getattr(args, key)).expanduser()
        setattr(args, key, str(p if p.is_absolute() else ROOT / p))
    p = Path(args.TEXT.DAMSM_NAME).expanduser()
    args.TEXT.DAMSM_NAME = str(p if p.is_absolute() else ROOT / p)
    if args.imsize != 256:
        raise ValueError('Paper configurations require imsize=256')
    if args.cond_dim != args.TEXT.EMBEDDING_DIM:
        raise ValueError('cond_dim must match TEXT.EMBEDDING_DIM')
    if args.batch_size < 2:
        raise ValueError('batch_size >= 2 is required for mismatched pairs')
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load(path, device='cpu'):
    # Checkpoints include optimizer/RNG Python objects; only load trusted local files.
    return torch.load(path, map_location=device, weights_only=False)


def encoder_and_vocab(args, device):
    with open(Path(args.data_dir) / 'captions_DAMSM.pickle', 'rb') as f:
        metadata = pickle.load(f, encoding='latin1')
    wordtoix = metadata[3]
    encoder = RNN_ENCODER(len(metadata[2]), nhidden=args.cond_dim).to(device)
    state = load(args.TEXT.DAMSM_NAME)
    encoder.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
    encoder.requires_grad_(False).eval()
    return encoder, wordtoix


def generator(args, device):
    return NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size,
                args.use_tf, args.use_cf).to(device)


def loader(args, split):
    dataset = ContourDataset(args, split)
    manifest = Path(args.contour_dir) / 'preprocess.json'
    if not manifest.exists():
        raise FileNotFoundError('Run preprocess before training/evaluation')
    meta = json.loads(manifest.read_text())
    if (meta['imsize'], meta['low'], meta['high']) != (args.imsize, args.canny_low, args.canny_high):
        raise ValueError('Contour preprocessing/config mismatch; run preprocess again')
    if split == 'train' and len(dataset) < args.batch_size:
        raise ValueError('Training dataset is smaller than batch_size')
    return DataLoader(dataset, args.batch_size, shuffle=split == 'train',
                      drop_last=split == 'train', num_workers=args.num_workers)


def gradient_penalty(images, sentences, scores, weight):
    gradients = torch.autograd.grad(scores.sum(), (images, sentences), create_graph=True)
    flat = torch.cat([g.flatten(1) for g in gradients], 1)
    return weight * flat.square().sum(1).pow(3).mean()


def train_step(g, d, c, optg, optd, images, sentence, contour, args, sobel):
    images = images.detach().requires_grad_(True)
    sentence = sentence.detach().requires_grad_(True)
    noise = torch.randn(len(images), args.z_dim, device=images.device)
    optd.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = g(noise, sentence, contour)
    features = d(images)
    real = c(features, sentence, contour)
    # Roll the paired text+contour condition together to create mismatched triples.
    mismatch = c(features, sentence.roll(1, 0), contour.roll(1, 0))
    negative = c(d(fake), sentence, contour)
    gp = gradient_penalty(images, sentence, real, args.magp_weight) if args.magp_weight else real.new_zeros(())
    loss_d = F.relu(1-real).mean() + .5*(F.relu(1+mismatch).mean()+F.relu(1+negative).mean()) + gp
    loss_d.backward()
    optd.step()
    optd.zero_grad(set_to_none=True)
    d.requires_grad_(False)
    c.requires_grad_(False)
    optg.zero_grad(set_to_none=True)
    fake = g(noise, sentence.detach(), contour)
    adv = -c(d(fake), sentence.detach(), contour).mean()
    structural = contour_loss(fake, contour, sobel, args.contour_reduction)
    loss_g = adv + args.lambda_contour * structural
    loss_g.backward()
    optg.step()
    d.requires_grad_(True)
    c.requires_grad_(True)
    return {k: float(v.detach()) for k, v in dict(d=loss_d, g=loss_g, adv=adv, contour=structural, magp=gp).items()}


def checkpoint(args, g, d, c, optg, optd, epoch, step):
    return dict(format='CTFGAN-v1', config=vars_for_json(args), epoch=epoch, step=step,
                model=dict(netG=g.state_dict(), netD=d.state_dict(), netC=c.state_dict()),
                optimizers=dict(G=optg.state_dict(), D=optd.state_dict()),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))


def vars_for_json(value):
    if isinstance(value, SimpleNamespace):
        return {k: vars_for_json(v) for k, v in vars(value).items()}
    return value


def load_generator(args, path, device):
    state = load(path)
    if state.get('format') != 'CTFGAN-v1':
        raise ValueError('Expected a trained CTFGAN checkpoint in CTFGAN-v1 format')
    for key in ('nf', 'z_dim', 'cond_dim', 'imsize', 'ch_size', 'use_tf', 'use_cf', 'dataset_name'):
        if state['config'][key] != getattr(args, key):
            raise ValueError(f'Checkpoint configuration mismatch: {key}')
    g = generator(args, device)
    g.load_state_dict(state['model']['netG'])
    return g.eval(), state


def train(args, device, resume=None, max_steps=None):
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    dl = loader(args, 'train')
    encoder, _ = encoder_and_vocab(args, device)
    g = generator(args, device)
    d, c = NetD(args.nf, args.imsize, args.ch_size).to(device), NetC(args.nf, args.cond_dim).to(device)
    optg = torch.optim.Adam(g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optd = torch.optim.Adam(list(d.parameters())+list(c.parameters()), lr=args.lr_d, betas=(args.beta1, args.beta2))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start, step = 1, 0
    if resume:
        _, state = load_generator(args, resume, device)
        if state.get('incomplete_epoch'):
            raise ValueError('Smoke checkpoints are for inference/debugging; resume from latest.pth')
        for key in ('lambda_contour', 'contour_reduction', 'lr_g', 'lr_d', 'beta1', 'beta2', 'batch_size', 'magp_weight', 'canny_low', 'canny_high'):
            if state['config'][key] != getattr(args, key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        for name, model in [('netG', g), ('netD', d), ('netC', c)]:
            model.load_state_dict(state['model'][name])
        optg.load_state_dict(state['optimizers']['G'])
        optd.load_state_dict(state['optimizers']['D'])
        start, step = state['epoch']+1, state['step']
        rng = state['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    (out / 'config.json').write_text(json.dumps(vars_for_json(args), indent=2))
    sobel = SobelContour(args.sobel_eps).to(device)
    writer = SummaryWriter(str(out / 'logs'))
    for epoch in range(start, args.max_epoch+1):
        g.train(); d.train(); c.train()
        progress = tqdm(dl, desc=f'Epoch {epoch}/{args.max_epoch}')
        for images, captions, lengths, contour, keys in progress:
            images, contour = images.to(device), contour.to(device)
            sentence = encode_text(encoder, captions, lengths)
            metrics = train_step(g, d, c, optg, optd, images, sentence, contour, args, sobel)
            step += 1
            for name, value in metrics.items():
                writer.add_scalar('loss/'+name, value, step)
            progress.set_postfix(metrics)
            if max_steps and step >= max_steps:
                # Mid-epoch smoke checkpoint is deliberately not resumable as a full epoch.
                smoke = checkpoint(args, g, d, c, optg, optd, epoch-1, step)
                smoke['incomplete_epoch'] = True
                torch.save(smoke, out/'smoke.pth')
                writer.close()
                return
        if epoch % args.gen_interval == 0:
            g.eval()
            with torch.no_grad():
                preview = g(torch.zeros(len(images), args.z_dim, device=device), sentence.detach(), contour)
                save_image(preview, str(out/f'epoch_{epoch:04d}.png'), normalize=True, value_range=(-1, 1))
        if epoch % args.save_interval == 0 or epoch == args.max_epoch:
            state = checkpoint(args, g, d, c, optg, optd, epoch, step)
            torch.save(state, out/f'epoch_{epoch:04d}.pth')
            torch.save(state, out/'latest.pth')
    writer.close()


def sample(args, device, checkpoint_path, manifest, count, seed):
    encoder, vocab = encoder_and_vocab(args, device)
    g, _ = load_generator(args, checkpoint_path, device)
    out = Path(args.output_dir)/'samples'
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    records = []
    for i, row in enumerate(rows):
        tokens = [vocab[t] for t in re.findall(r'\w+', row['text'].lower()) if t in vocab][:args.TEXT.WORDS_NUM]
        if not tokens:
            raise ValueError(f'Manifest row {i+1}: no known vocabulary words')
        sentence = encode_text(encoder, torch.tensor([tokens]), torch.tensor([len(tokens)]))
        path = Path(row['contour']).expanduser()
        if not path.is_absolute():
            path = Path(manifest).resolve().parent/path
        with Image.open(path) as image:
            image = image.convert('L').resize((args.imsize, args.imsize), Image.Resampling.NEAREST)
            contour = torch.from_numpy(np.asarray(image).copy()).float()[None, None].to(device)/255
        for j in range(count):
            sample_seed = seed+i*count+j
            seed_all(sample_seed)
            with torch.no_grad():
                fake = g(torch.randn(1, args.z_dim, device=device), sentence, contour)
            name = f'{i:06d}_{j:02d}.png'
            save_image(fake, str(out/name), normalize=True, value_range=(-1, 1))
            records.append(dict(image=name, text=row['text'], contour=str(path), seed=sample_seed))
    (out/'manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))


def evaluate(args, device, checkpoint_path, repeats, seed):
    """Export paired test samples; FID uses the baseline's torchvision Inception."""
    from lib.modules import calculate_frechet_distance
    from models.inception import InceptionV3
    dl = loader(args, 'test')
    encoder, _ = encoder_and_vocab(args, device)
    g, _ = load_generator(args, checkpoint_path, device)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    stats = []
    out = Path(args.output_dir)/'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    records, index = [], 0
    from tqdm import tqdm
    for repeat in range(repeats):
        for images, captions, lengths, contour, keys in tqdm(dl, desc=f'Evaluation {repeat+1}'):
            sentence = encode_text(encoder, captions, lengths)
            noise = []
            for j in range(len(images)):
                seed_all(seed+index+j)
                noise.append(torch.randn(args.z_dim, device=device))
            with torch.no_grad():
                fake = g(torch.stack(noise), sentence, contour.to(device))
                features = model((fake+1)/2)[0].flatten(1)
            stats.append(features.cpu().numpy())
            for j, key in enumerate(keys):
                name = f'{index+j:06d}.png'
                save_image(fake[j], str(out/name), normalize=True, value_range=(-1, 1))
                records.append(dict(image=name, key=key, repeat=repeat, seed=seed+index+j))
            index += len(images)
    features = np.concatenate(stats)
    if len(features) < 2:
        raise ValueError('FID requires at least two samples')
    npz_path = Path(args.npz_path)
    if not npz_path.is_absolute():
        npz_path = ROOT/npz_path
    with np.load(npz_path) as reference:
        fid = calculate_frechet_distance(reference['mu'], reference['sigma'], features.mean(0), np.cov(features, rowvar=False))
    result = dict(FID=float(fid), samples=index, seed=seed, repeats=repeats,
                  caption_policy='first caption per image', noise='standard normal')
    (out/'metrics.json').write_text(json.dumps(result, indent=2))
    (out/'manifest.json').write_text(json.dumps(records, indent=2))
    print(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preprocess', 'train', 'sample', 'evaluate'])
    parser.add_argument('--cfg', default=str(ROOT/'code/cfg/ctfgan_bird.yml'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int)
    cli = parser.parse_args(argv)
    args = config(cli.cfg, dict(batch_size=cli.batch_size, num_workers=cli.num_workers,
                                output_dir=cli.output_dir, max_epoch=cli.epochs))
    if cli.command == 'preprocess':
        preprocess(args)
        return
    if cli.count < 1 or cli.repeats < 1:
        parser.error('--count and --repeats must be positive')
    device = torch.device(cli.device or args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: install CUDA PyTorch, or pass --device cpu for smoke tests')
    seed_all(args.manual_seed)
    if cli.command == 'train':
        train(args, device, cli.resume, cli.max_steps)
    else:
        if not cli.checkpoint:
            parser.error('--checkpoint is required')
        if cli.command == 'sample':
            if not cli.manifest:
                parser.error('--manifest is required: JSONL text/contour pairs')
            sample(args, device, cli.checkpoint, cli.manifest, cli.count, cli.seed)
        else:
            evaluate(args, device, cli.checkpoint, cli.repeats, cli.seed)


if __name__ == '__main__':
    main()
