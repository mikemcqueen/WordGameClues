{
  'targets': [{
    'target_name': 'experiment',
    'type': 'none',
    'dependencies': [ 'build_experiment' ],
    'actions': [{
      'action_name': 'copy_plugin',
      'inputs': [ '<(PRODUCT_DIR)/experiment.node' ],
      'outputs': [ '<(PRODUCT_DIR)/..' ],
      'action': [ 'sh', '-c', 'cp <(_inputs) <(_outputs)' ]
    }]
  },
  {
    'target_name': 'build_experiment',
    'product_name': 'experiment',
    'cflags_cc': [ '-fPIC -std=c++23 -Wno-unused-function -O3 -Wall -pedantic -Werror -I../wtf-threadpool' ],
    'cflags!': [ '-fno-exceptions' ],
    'cflags_cc!': [ '-fno-exceptions' ],
    'ldflags': [ '-Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib' ],
    'sources': [
      'candidates.cpp',
      'clue-manager.cpp',
      'cm-precompute.cpp',
      'components.cpp',
      'cuda-common.cpp',
      'filter-support.cpp',
      'index.cpp',
      'merge-support.cpp',
      'validator.cpp',
      'unwrap.cpp',
      'wrap.cpp'
    ],
    'include_dirs': [
      '<!@(node -p \'require("node-addon-api").include\')',
      '/usr/local/cuda/include'
    ],
    'library_dirs': [
      '/usr/local/cuda/lib64'
    ],
    'libraries': [ '-lcudart', '-lcudadevrt' ],
    'dependencies': [ 'kernels' ],
    'defines': [ 'NAPI_CPP_EXCEPTIONS' ]
  },
  {
    'target_name': 'kernels',
    'type': 'static_library',
    'sources': [
       'merge.cu',
       'filter.cu'
    ],
    'rules': [{
      'extension': 'cu',
      'rule_name': 'kernel lib',
      'message': 'CUDA static lib',
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'outputs': [
        '<(SHARED_INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o',
        '<(SHARED_INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT)_dlink.o'
       ],
      'process_outputs_as_sources': 1,
      'action': []
    }],
    'dependencies': [ 'build_kernels' ]
  },
  {
    'target_name': 'build_kernels',
    'type': 'none',
    'sources': [
        'filter.cu',
        'merge.cu'
    ],
    'rules': [{
      'extension': 'cu',           
      'rule_name': 'compile kernel',
      'message': 'compile and device link CUDA file on linux',
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'outputs': [ '<(RULE_INPUT_ROOT)_dummy_output' ],
      'action': [
          'env', 'DIR=<(SHARED_INTERMEDIATE_DIR)', 'FILE=<(RULE_INPUT_ROOT)',
          'make', '-sf', 'kernel.mk', 'build'
      ]
    }]
  }]
}
