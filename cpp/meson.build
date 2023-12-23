project(
  'dgo',
  'cpp',
  version : '0.0.1',
  default_options : [
    'cpp_std=c++20',
  ]
)

gtest_proj = subproject('gtest')
gtest      = gtest_proj.get_variable('gtest_main_dep')

cc             = meson.get_compiler('cpp')
torch_dep      = dependency('Torch', method: 'cmake')
torch_cpu_dep  = cc.find_library(
  'torch_cpu',
  dirs: join_paths(meson.current_source_dir(), 'subprojects\\libtorch\\lib'),
  required: true
)
c10_dep        = cc.find_library(
  'c10',
  dirs: join_paths(meson.current_source_dir(), 'subprojects\\libtorch\\lib'),
  required: true
)

##################
# Command line interface
##################

cli = executable(
  'cli',
  'cmd/cli.cpp',
)
test('cli', cli)


##################
# Benchmark for Algorithms
##################

benchmark_alphazero = executable(
  'benchmark_alphazero',
  'cmd/benchmark_alphazero.cpp',
  dependencies : [torch_dep, torch_cpu_dep, c10_dep],
)


##################
# Tests for Core
##################

binary_print_test = executable(
  'binary_print_test',
  'cmd/test_binary_print.cpp',
  dependencies: gtest
)
test('binary_print', binary_print_test)

lemke_howson_test = executable(
  'lemke_howson_test',
  'cmd/test_lemke_howson.cpp',
  dependencies: gtest
)
test('lemke_howson', lemke_howson_test)


##################
# Tests for third_party
##################

libtorch_cpu_basic_test = executable(
  'libtorch_cpu_basic_test',
  'cmd/test_libtorch_cpu_basic.cpp',
  dependencies : [gtest, torch_dep, torch_cpu_dep, c10_dep],
)
test('libtorch_cpu_basic', libtorch_cpu_basic_test)


##################
# Tests for examples
##################

game_quoridor_test = executable(
  'game_quoridor_test',
  'cmd/test_game_quoridor.cpp',
  dependencies : [gtest, torch_dep, torch_cpu_dep, c10_dep],
)
test('game_quoridor', game_quoridor_test, workdir : meson.source_root())


##################
# Tests for algorithms
##################

algorithm_alphazero_mcts_node_test = executable(
  'algorithm_alphazero_mcts_node_test',
  'cmd/test_alphazero_mcts_node.cpp',
  dependencies : [gtest, torch_dep, torch_cpu_dep, c10_dep],
)
test('algorithm_alphazero_mcts_node', algorithm_alphazero_mcts_node_test, workdir : meson.source_root(), timeout : 300)


##################
# Other executables
##################

selfplay_quoridor = executable(
  'selfplay_quoridor',
  'misc/alphazero_selfplay_quoridor.cpp',
  dependencies : [torch_dep, torch_cpu_dep, c10_dep],
)
