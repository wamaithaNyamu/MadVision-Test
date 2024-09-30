<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]






<!-- ABOUT THE PROJECT -->
## About The Project

<p>
Let's say you publish longform videos on YouTube and want to break them down into shortform content for other platforms such as TikTok or Instagram Reels. This project implements an event drive architecture that uses, RabbitMQ Celery, OpenAI LLM, whisper, FastAPI and Supabase. RabbitMQ is used as the broker while celery is used as the worker. OpenAI is used as the LLM that helps get insighful clips from the transcription we get back from Whisper. All the information is stored on Supabase.
</p>

<img src="Media/arch.png" alt="Product Name Screen Shot" width="60%" />



<!-- GETTING STARTED -->
## Getting Started

To get started you need to have an [OpenAI developer account](https://platform.openai.com/). You also need a Supabase account. Follow this instructions to create a [supabase](https://supabase.com/) table and give it the name `videos`.

### Prerequisites

To run this project you need the following:
- Python 3.12+
- Docker installed 
- Pip installed (comes with Python 3.12 by default)
- Venv (optional)


### Installation
1. Get API key from [OpenAI](https://platform.openai.com/docs/quickstart) and Supabase [credentials](https://supabase.com/dashboard).

2. Clone the repo
   ```sh
   git clone https://github.com/wamaithanyamu/MadVision-Test.git
   ```
3. Install modules
   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add the following
   ```shell
    OPENAI_API_KEY=
    SUPABASE_URL=
    SUPABASE_KEY=
    SUpabase_passdb=
    Supabase_Service_Key=
    VIDEOS_FOLDER_NAME=videos_output
    CLIPS_FOLDER_NAME=clips_output
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin <your-github-username/repo-name>
   git remote -v # confirm the changes
   ```

6. Install ffmpeg (MacOS)
```shell
brew install ffmpeg
```

7. Run rabbit mq from docker
```shell
sudo docker compose up
```

8. Give permissions to all bash scripts

```shell
chmod +x *.sh
```
9. On a separate terminal run consumers (RabbitMQ)
```shell
./run_consumers.py
```

10. On a separate terminal run celery worker

```shell
./run_celery

```

11. Run Fast API in another terminal window

```shell
fastapi dev app.py
```

Navigate to https://localhost:8000/docs to interact with the API


<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/wamaithanyamu/MadVision-Test/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=wamaithanyamu/MadVision-Test" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@wamaithanyamu](https://twitter.com/wamaithanyamu) - hello@wamaithanyamu.com

Project Link: [https://github.com/wamaithanyamu/MadVision-Test](https://github.com/wamaithanyamu/MadVision-Test)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/wamaithanyamu/MadVision-Test.svg?style=for-the-badge
[contributors-url]: https://github.com/wamaithanyamu/MadVision-Test/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wamaithanyamu/MadVision-Test.svg?style=for-the-badge
[forks-url]: https://github.com/wamaithanyamu/MadVision-Test/network/members
[stars-shield]: https://img.shields.io/github/stars/wamaithanyamu/MadVision-Test.svg?style=for-the-badge
[stars-url]: https://github.com/wamaithanyamu/MadVision-Test/stargazers
[issues-shield]: https://img.shields.io/github/issues/wamaithanyamu/MadVision-Test.svg?style=for-the-badge
[issues-url]: https://github.com/wamaithanyamu/MadVision-Test/issues
[license-shield]: https://img.shields.io/github/license/wamaithanyamu/MadVision-Test.svg?style=for-the-badge
[license-url]: https://github.com/wamaithanyamu/MadVision-Test/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/wamaithanyamu
[product-screenshot]: Media/arch.png



