{
  "targets": [
    {
      "target_name": "experiment",
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "sources": [
        "candidates.cpp",
        "cm-precompute.cpp",
        "combo-maker.cpp",
        "greeting.cpp",
        "index.cpp",
        "wrap.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "defines": [ "NAPI_CPP_EXCEPTIONS" ]
    }
  ]
}