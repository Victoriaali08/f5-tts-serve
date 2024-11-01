# f5-tts-serve

A simple wrapper around [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://github.com/SWivid/F5-TTS) that provides an OpenAI-compatible API endpoint for speech generation (`/v1/audio/speech`).

This is just a toy, a POC, not suitable for production or multi-user use.

Yes, I'm aware of [openedai-speech](https://github.com/matatonic/openedai-speech/) and I was originally going to build on top of that project. But I was curious how easy or hard it would be to write a `/v1/audio/speech` endpoint provider from scratch.

## Features

* Provides an OpenAI-compatible API endpoint for `/v1/audio/speech`

   My testing consisted of using `curl`, Open-WebUI, and SillyTavern as frontends.

* Multiple voices are supported. Note: At the moment, the `model` parameter (which is usually given the value `tts-1` or `tts-1-hd`) is totally ignored. Only `voice` matters.

* Can also specify the speaking rate

* The supported audio formats are based on the formats supported by the [soundfile](https://pypi.org/project/soundfile/) Python package, which itself seems to be based off of [libsndfile](http://www.mega-nerd.com/libsndfile/).

   Basically, this means every format listed in the OpenAI API spec *except* for AAC should be supported:

   * WAV
   * MP3
   * FLAC
   * OPUS (in .ogg container)
   * PCM (signed 16-bit little-endian)

### ToDo

* [ ] Dockerfile
* [ ] More robustness, especially in code that executes in background threads
* [ ] API key support?
* [ ] AAC support? Are there any frontends that demand AAC?

## Installation

At the moment, I'm including the F5-TTS project as a git submodule. (Whether or not this is a bad idea, I guess I'll see.)

    git clone --recurse-submodules https://github.com/asaddi/f5-tts-serve.git

Make a copy of the config:

    cp config.yaml.default config.yaml

To add voices, copy a reference WAV file somewhere (like the `voices` directory) and edit `config.yaml`. See the comments there. (The included voice, `basic_ref_en.wav`, originated from F5-TTS.)

Create a venv/virtualenv (or use conda) and then install the requirements:

    pip install -r requirements.txt

If you'd like to use something other than CUDA 12.4, use the unpinned requirements and specify `--extra-index-url`, for example:

    pip install -r requirements.in --extra-index-url https://download.pytorch.org/whl/cu118

## Running

With your venv/virtualenv/conda environment active:

    python server.py

By default, it will listen to `127.0.0.1` port 8000. You can change this by adding the `--host` and `--port` arguments to the above.

### Example Client Usage (using curl)

    curl -o test.wav http://localhost:8000/v1/audio/speech \
      -H "Content-Type: application/json" \
      -d '{
      "model": "tts-1",
      "input": "That quick beige fox jumped in the air over each thin dog. Look out, I shout, for he'\''s foiled you again, creating chaos.",
      "voice": "basic",
      "response_format": "wav"
    }'

The parameter `response_format`, if omitted, defaults to `mp3`.

The API documentation can be found [here](https://platform.openai.com/docs/api-reference/audio).

## License

Licensed under [the MIT license](https://opensource.org/license/mit).
