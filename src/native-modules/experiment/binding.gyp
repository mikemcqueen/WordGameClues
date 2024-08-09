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
      'stream-data.cpp',
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
    'dependencies': [ 'kernels_lib' ],
    'defines': [ 'NAPI_CPP_EXCEPTIONS' ]
  },
  {
    'target_name': 'kernels_lib',
    'type': 'static_library',
    'sources': [
       '<(SHARED_INTERMEDIATE_DIR)/merge.o',
       '<(SHARED_INTERMEDIATE_DIR)/filter.o',
       '<(SHARED_INTERMEDIATE_DIR)/or-filter.o',
       '<(SHARED_INTERMEDIATE_DIR)/kernels_dlink.o',
    ],
    'rules': [{
      'extension': 'o',
      'rule_name': 'device link kernels',
      'message': 'device link kernels on linux',
      'variables': {
        'obj_files': [
          'merge.o',
          'filter.o',
          'or-filter.o',
        ]
      },
      'inputs': [
      ],
      'outputs': [ '<(SHARED_INTERMEDIATE_DIR)/kernels_dlink.o' ],
      'process_outputs_as_sources': 1,
      'action': [
        'env', 'DIR=<(SHARED_INTERMEDIATE_DIR)',
               'FILE=kernels_dlink.o',
               'OBJ_FILES=<@(obj_files)',
        'make', '-sf', 'kernel.mk', 'dlink'
      ]
    }], # end rules
  },
  {
    'target_name': 'compile_kernels',
    'type': 'none',
    'sources': [
        'merge.cu',
        'or-filter.cu',
        'filter.cu',
    ],
    'rules': [{
      'extension': 'cu',           
      'rule_name': 'compile kernels',
      'message': 'compile CUDA kernels on linux',
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'outputs': [ '<(SHARED_INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o' ],
      'action': [
          'env', 'DIR=<(SHARED_INTERMEDIATE_DIR)', 'FILE=<(RULE_INPUT_ROOT)',
          'make', '-isf', 'kernel.mk', 'compile'
      ]
    }]
  }] # end targets
}
