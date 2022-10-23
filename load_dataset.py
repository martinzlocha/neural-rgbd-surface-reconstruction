from load_scannet import load_scannet_data
from load_record3d import load_record3d_data


def load_dataset(args):

    if args.dataset_type == "scannet":
        images, depth_images, poses, hwf, frame_indices = load_scannet_data(basedir=args.datadir,
                                                                            trainskip=args.trainskip,
                                                                            downsample_factor=args.factor,
                                                                            translation=args.translation,
                                                                            sc_factor=args.sc_factor,
                                                                            crop=args.crop)

        print('Loaded scannet', images.shape, hwf, args.datadir)

    elif args.dataset_type == "record3d":

        images, depth_images, poses, hwf, frame_indices = load_record3d_data(basedir=args.datadir,
                                                                            trainskip=args.trainskip,
                                                                            downsample_factor=args.factor,
                                                                            translation=args.translation,
                                                                            sc_factor=args.sc_factor,
                                                                            crop=args.crop)

        print('Loaded Record3D', images.shape, hwf, args.datadir)

    # Calls to other dataloaders go here
    # elif args.dataset_type == "":

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    return images, depth_images, poses, hwf, frame_indices
