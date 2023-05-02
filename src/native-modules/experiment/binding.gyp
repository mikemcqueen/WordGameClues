{
  "targets": [
    {
      "target_name": "experiment",
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "sources": [
        "index.cpp",
        "greeting.cpp",
        "combo-maker.cpp",
        "cm-precompute.cpp",
	"wrap.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "defines": [ "NAPI_CPP_EXCEPTIONS" ]
    }
  ]
}