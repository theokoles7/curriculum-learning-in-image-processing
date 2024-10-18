__all__ = ["wavelet"]

from curriculums.compression        import rmse
from curriculums.edge_density       import edge_density
from curriculums.spatial_frequency  import spatial_frequency
from curriculums.wavelet            import wavelet_energy, wavelet_entropy

from curriculums.batch_sampler      import CurriculumSampler

curriculums:    dict =  {
    "edge_density":         edge_density,
    "rmse":                 rmse,
    "spatial_frequency":    spatial_frequency,
    "wavelet_energy":       wavelet_energy,
    "wavelet_entropy":      wavelet_entropy
}