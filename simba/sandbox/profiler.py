import cProfile
import App


profiler = cProfile.Profile()
profiler.enable()
App
profiler.disable()
profiler.print_stats()