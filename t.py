import subprocess

wave2lipCheckpointPath = "checkpoints/wav2lip.pth"
wav2lipFolderName = r"E:\chat\Wav2Lip-master"

inputAudioPath = r"E:\chat\inputs\ms.wav"
inputVideoPath = r"E:\chat\inputs\salma720.mp4"
lipSyncedOutputPath = r'E:\chat\outputs\out.mp4'

def _wave2lip():
        """ Run wave2lip model
        """

        command = [
            "python", "inference_yolo.py",
            "--checkpoint_path", wave2lipCheckpointPath,
            "--face", inputVideoPath,
            "--audio", inputAudioPath,
            "--outfile", lipSyncedOutputPath
        ]
        try:
            with subprocess.Popen(command, cwd=wav2lipFolderName, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as process:
                for line in process.stdout:
                    print(line, end='')

            print("Wav2Lip execution completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")


_wave2lip()