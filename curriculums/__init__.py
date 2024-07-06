__all__ = ["wavelet"]

from curriculums.compression        import rmse
from curriculums.spatial_frequency  import spatial_frequency
from curriculums.wavelet            import wavelet_energy, wavelet_entropy

from curriculums.batch_sampler      import CurriculumSampler

curriculums:    dict =  {
    "rmse":                 rmse,
    "spatial_frequency":    spatial_frequency,
    "wavelet_energy":       wavelet_energy,
    "wavelet_entropy":      wavelet_entropy
}