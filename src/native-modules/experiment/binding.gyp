{
  'targets': [{
    'target_name': 'experiment',
    'cflags_cc': [ '-fPIC -std=c++20 -Wno-unused-function' ],
    'cflags!': [ '-fno-exceptions' ],
    'cflags_cc!': [ '-fno-exceptions' ],
    'ldflags': [ '-Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib' ],
    'sources': [
      'candidates.cpp',
      'cm-precompute.cpp',
      'filter-support.cpp',
      'index.cpp',
      'merge-support.cpp',
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
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'rule_name': 'CUDA static lib',
      'message': 'CUDA static lib',
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
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'rule_name': 'compile CUDA on linux',
      'message': 'compile and device link CUDA file on linux',
      'outputs': [ '<(RULE_INPUT_ROOT)_dummy_output' ],
      'action': [
          'env', 'DIR=<(SHARED_INTERMEDIATE_DIR)', 'FILE=<(RULE_INPUT_ROOT)',
          'make', '-f', 'kernel.mk'
      ]
    }]
  }]
}
