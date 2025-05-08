from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Custom_NPY
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'ExerCube': Dataset_Custom_NPY,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        # make sure you've set args.data = 'custom_npy'
        # and args.data_paths = {'train':..., 'val':..., 'test':...}
        data_set = Data(
            args       = args,
            flag       = flag,
            size       = [args.seq_len, args.seq_len, args.seq_len],
            data_paths = args.data_paths,
            scale      = True
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size   = batch_size,
            shuffle      = shuffle_flag,
            num_workers  = args.num_workers,
            drop_last    = drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        # ── NPY‐based loader for ExerCube ──
        if args.data == 'ExerCube':
            data_set = Data(
                args       = args,
                flag       = flag,
                size       = [args.seq_len, args.label_len, args.pred_len],
                data_paths = args.data_paths,
                scale      = True
            )
        # ── “Old” CSV & built-in datasets ──
        else:
            data_set = Data(
                args              = args,
                root_path         = args.root_path,
                data_path         = args.data_path,
                flag              = flag,
                size              = [args.seq_len, args.label_len, args.pred_len],
                features          = args.features,
                target            = args.target,
                timeenc           = timeenc,
                freq              = freq,
                seasonal_patterns = args.seasonal_patterns
            )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size   = args.batch_size,
            shuffle      = False if flag.lower()=='test' else True,
            num_workers  = args.num_workers,
            drop_last    = False
        )
        return data_set, data_loader