import itertools

from gcsfs.tests.perf.microbenchmarks.glob.parameters import GlobBenchmarkParameters
from gcsfs.tests.perf.microbenchmarks.listing.configs import ListingConfigurator


class GlobConfigurator(ListingConfigurator):
    param_class = GlobBenchmarkParameters


def get_glob_benchmark_cases():
    return GlobConfigurator(__file__).generate_cases()
