from typing import Literal
from .tabular import TabularEncoder
from .wide_resnet import WideResNet
from .resnet import ResNet, ResNetDecoder
from .pretrained import tv_resnet18, tv_resnet50, tv_efficientnet_v2_s, tv_swin_t


EncoderType = Literal[
    "tabular", 
    "resnet18", 
    "wide-resnet",
    "resnet18-tv",
    "resnet50",
    "efficientnet_v2_s",
    "swin_t",
]


__all__ = [
    "EncoderType",
    "TabularEncoder",
    "WideResNet",
    "ResNet",
    "ResNetDecoder",
    "tv_resnet18",
    "tv_resnet50",
    "tv_efficientnet_v2_s",
    "tv_swin_t",
    "get_pretrained_path",
]


def get_pretrained_path(encoder, seed, latent_dim, spectral, dataset_name, coeff=0, reconst_weight=0, residual=True):
    # CIFAR100 Pretrained encoders
    if dataset_name == "cifar100":
        pretrained_resnet18 = {
            42: {
                16: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4614226/4614226.pt',
                32: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4108603/4108603.pt',
                64: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3744854/3744854.pt',
                128: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2341057/2341057.pt',
                256: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1719583/1719583.pt',
            },
            43: {
                16: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2414798/2414798.pt',
                32: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7760003/7760003.pt',
                64: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1615384/1615384.pt',
                128: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6205509/6205509.pt',
                256: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7603253/7603253.pt',
            },
            44: {
                16: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2963957/2963957.pt',
                32: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6368108/6368108.pt',
                64: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3777549/3777549.pt',
                128: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4858146/4858146.pt',
                256: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-488170/488170.pt',
            },
            45: {
                16: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8187754/8187754.pt',
                32: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4322827/4322827.pt',
                64: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1373707/1373707.pt',
                128: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7007695/7007695.pt',
                256: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5086719/5086719.pt',
            },
            46: {
                16: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6710792/6710792.pt',
                32: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-670446/670446.pt',
                64: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9875407/9875407.pt',
                128: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9846584/9846584.pt',
                256: '/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3834251/3834251.pt',
            }
        }
        pretrained_resnet18_no_residual = {
            42: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6210606/6210606.pt",
            43: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-840492/840492.pt",
            44: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8110957/8110957.pt",
            45: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2821078/2821078.pt",
            46: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6615310/6615310.pt",
        }
        pretrained_resnet18_spectral = {
            42: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6377459/6377459.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5708456/5708456.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4437923/4437923.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-728977/728977.pt",
                6: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7707870/7707870.pt",
            },
            43: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7361662/7361662.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2694682/2694682.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-820737/820737.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9718502/9718502.pt",
                6: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1026944/1026944.pt",
            },
            44: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7610642/7610642.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4743415/4743415.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-587258/587258.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5780607/5780607.pt",
                6: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6473972/6473972.pt",
            },
            45: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9114582/9114582.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2395878/2395878.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4188785/4188785.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8388082/8388082.pt",
                6: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7113975/7113975.pt",
            },
            46: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9769144/9769144.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8031997/8031997.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3082722/3082722.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6440122/6440122.pt",
                6: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5062255/5062255.pt",
            },
        }
        pretrained_resnet18_reconst = {
            42: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7038374/7038374.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3698379/3698379.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7536477/7536477.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4855124/4855124.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1338687/1338687.pt",
            },
            43: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-968349/968349.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1800268/1800268.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2089632/2089632.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7674252/7674252.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5557872/5557872.pt",
            },
            44: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6348944/6348944.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9915287/9915287.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6795573/6795573.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-634741/634741.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5513879/5513879.pt",
            },
            45: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2265111/2265111.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4556446/4556446.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2585757/2585757.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6449461/6449461.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1037745/1037745.pt",
            },
            46: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9205088/9205088.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4954678/4954678.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8854500/8854500.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3069815/3069815.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6137963/6137963.pt",
            },
        }

    # CIFAR10 Pretrained encoders
    elif dataset_name == "cifar10":
        pretrained_resnet18 = {
            42: {
                8: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8375710/8375710.pt",
                16: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6637601/6637601.pt",
                32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7698256/7698256.pt",
                64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2396987/2396987.pt",
                128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4443951/4443951.pt",
            },
            43: {
                8: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2458992/2458992.pt",
                16: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-42269/42269.pt",
                32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1774292/1774292.pt",
                64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2114501/2114501.pt",
                128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2085740/2085740.pt",
            },
            44: {
                8: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2393814/2393814.pt",
                16: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3010507/3010507.pt",
                32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-90543/90543.pt",
                64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1057974/1057974.pt",
                128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4927443/4927443.pt",
            },
            45: {
                8: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1057696/1057696.pt",
                16: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1537122/1537122.pt",
                32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1084952/1084952.pt",
                64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2219186/2219186.pt",
                128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6580286/6580286.pt",
            },
            46: {
                8: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2498986/2498986.pt",
                16: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6432462/6432462.pt",
                32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5228188/5228188.pt",
                64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6372029/6372029.pt",
                128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2214264/2214264.pt",
            },
        }
        pretrained_resnet18_no_residual = {
            42: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6073292/6073292.pt",
            43: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1462077/1462077.pt",
            44: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4687115/4687115.pt",
            45: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8816793/8816793.pt",
            46: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8568065/8568065.pt",
        }
        pretrained_resnet18_spectral = {
            42: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9807725/9807725.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7187926/7187926.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9790057/9790057.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6700828/6700828.pt",
            },
            43: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4345064/4345064.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7654344/7654344.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5704445/5704445.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6469209/6469209.pt",
            },
            44: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2992986/2992986.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8786237/8786237.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3231774/3231774.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1019771/1019771.pt",
            },
            45: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6411781/6411781.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6748383/6748383.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6327957/6327957.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2732319/2732319.pt",
            },
            46: {
                2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2030202/2030202.pt",
                3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6335556/6335556.pt",
                4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5357647/5357647.pt",
                5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4648119/4648119.pt",
            },
        }
        pretrained_resnet18_reconst = {
            42: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2342608/2342608.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4137722/4137722.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9418194/9418194.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9042538/9042538.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4408072/4408072.pt",
            },
            43: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-286084/286084.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5664676/5664676.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1988790/1988790.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8931239/8931239.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8356921/8356921.pt",
            },
            44: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4564226/4564226.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-113065/113065.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1248790/1248790.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7302278/7302278.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4595487/4595487.pt",
            },
            45: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2746270/2746270.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1387585/1387585.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7495800/7495800.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6524522/6524522.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4653994/4653994.pt",
            },
            46: {
                0.1: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6385039/6385039.pt",
                0.5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-687912/687912.pt",
                1.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1827612/1827612.pt",
                2.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-886556/886556.pt",
                4.0: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3341808/3341808.pt",
            },
        }

    # Camelyon Pretrained encoders
    pretrained_wide = {
        42: {
            32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4661907/4661907.pt",
            64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2608513/2608513.pt",
            128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3612365/3612365.pt",
            256: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5647119/5647119.pt",
            512: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1714803/1714803.pt",
        },
        43: {
            32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2159299/2159299.pt",
            64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6490296/6490296.pt",
            128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8500612/8500612.pt",
            256: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4562329/4562329.pt",
            512: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9649565/9649565.pt",
        },
        44: {
            32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8573393/8573393.pt",
            64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1175523/1175523.pt",
            128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9875384/9875384.pt",
            256: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1587586/1587586.pt",
            512: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4529874/4529874.pt",
        },
        45: {
            32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4266152/4266152.pt",
            64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5511408/5511408.pt",
            128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6269311/6269311.pt",
            256: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3517661/3517661.pt",
            512: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8265802/8265802.pt",
        },
        46: {
            32: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2572257/2572257.pt",
            64: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9978093/9978093.pt",
            128: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5951523/5951523.pt",
            256: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1789605/1789605.pt",
            512: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1158156/1158156.pt",
        },
    }
    pretrained_wide_no_residual = {
        42: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3679591/3679591.pt",
        43: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-560511/560511.pt",
        44: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8105827/8105827.pt",
        45: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-504417/504417.pt",
        46: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5284273/5284273.pt",
    }
    pretrained_wide_spectral = {
        42: {
            2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4663623/4663623.pt",
            3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6022674/6022674.pt",
            4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7606962/7606962.pt",
            5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6120868/6120868.pt",
        },
        43: {
            2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2798746/2798746.pt",
            3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-6323755/6323755.pt",
            4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2119600/2119600.pt",
            5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1344719/1344719.pt",
        },
        44: {
            2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9898114/9898114.pt",
            3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1807189/1807189.pt",
            4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9068578/9068578.pt",
            5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2526428/2526428.pt",
        },
        45: {
            2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-8733379/8733379.pt",
            3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2616473/2616473.pt",
            4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4839614/4839614.pt",
            5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-5339262/5339262.pt",
        },
        46: {
            2: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-2878329/2878329.pt",
            3: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7724064/7724064.pt",
            4: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9103470/9103470.pt",
            5: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3178867/3178867.pt",
        },
    }

    # # Pretrained on CIFAR100-224
    # pretrained_resnet50 = {
    #     42: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3823498/3823498.pt",
    #     43: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9804220/9804220.pt",
    #     44: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-825518/825518.pt",
    #     45: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-9300883/9300883.pt",
    #     46: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1942622/1942622.pt",
    # }
    # # Pretrained on CIFAR100-224-partial
    # pretrained_resnet50 = {
    #     42: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-3226067/3226067.pt",
    #     43: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-1728736/1728736.pt",
    #     44: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4160156/4160156.pt",
    #     45: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-7641457/7641457.pt",
    #     46: "/nfs/homedirs/zch/natpn-improve/src/pretrained/pretrained-latent-4763630/4763630.pt",
    # }
    # if encoder == "resnet50":
    #     return pretrained_resnet50[seed]

    if residual == False:
        if encoder == "resnet18":
            return pretrained_resnet18_no_residual[seed]
        elif encoder == "wide-resnet":
            return pretrained_wide_no_residual[seed]

    if reconst_weight > 0:
        if encoder == "resnet18":
            return pretrained_resnet18_reconst[seed][reconst_weight]

    if spectral == False:
        if encoder == "resnet18":
            return pretrained_resnet18[seed][latent_dim]
        elif encoder == "wide-resnet":
            return pretrained_wide[seed][latent_dim]
    else:
        if encoder == "resnet18":
            return pretrained_resnet18_spectral[seed][coeff]
        elif encoder == "wide-resnet":
            return pretrained_wide_spectral[seed][coeff]

    return ""
