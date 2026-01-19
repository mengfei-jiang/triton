import os
import subprocess
import sys
from pathlib import Path

# ======================== Configuration (Modify per your requirements) ========================
# 1. Basic Configuration
LLVM_REPO_URL = "https://github.com/llvm/llvm-project.git"
LLVM_LOCAL_REPO_URL = "https://github.com/mengfei-jiang/llvm-project.git"
LLVM_LOCAL_REPO_DIR = "./llvm-project"
TRITON_REPO_URL = "https://github.com/triton-lang/triton.git"
TRITON_LOCAL_REPO_URL = "https://github.com/mengfei-jiang/triton.git"
TRITON_LOCAL_REPO_DIR = "./triton"
REMOTE_NAME = "rocm"
REMOTE_URL = "https://github.com/others/others-repo.git"

# 2. Cherry-pick Configuration 
LLVM_CHERRY_PICK_COMMITS = "0df45af^..29323d1"
TRITON_CHERRY_PICK_COMMITS = "fbceeac^..5deb860"

# 3. Shell Test Script Configuration
SHELL_SCRIPT_PATH = "./triton/bench.sh"  # Path to the Shell test script
SHELL_SCRIPT_NAME = "./bench.sh"  # Path to the Shell test script

LLVM_CMAKE_CMD = [
    "cmake",
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "../llvm",
    "-DLLVM_ENABLE_PROJECTS=mlir;llvm;lld",
    "-DLLVM_TARGETS_TO_BUILD=host;NVPTX;AMDGPU"
]
NINJA_BUILD_CMD = f"ninja -j{os.cpu_count()}"  # Use all CPU cores

# ======================== Utility Functions (No modification needed) ========================
def run_cmd(cmd, cwd=None, env=None, timeout=300, capture_output=True):
    """
    Execute system command and return execution result
    :param cmd: Command (list format, e.g., ["git", "clone", "url"])
    :param cwd: Working directory for command execution
    :param env: Custom environment variables (dict)
    :param timeout: Timeout in seconds
    :param capture_output: Whether to capture stdout/stderr
    :return: (returncode, stdout, stderr)
    """
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)

    try:
        print(f"\n[Executing Command]: {' '.join(cmd)}")
        if capture_output:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=cmd_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            return result.returncode, stdout, stderr
        else:
            # Print output in real-time (suitable for time-consuming commands like git clone)
            process = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line.strip())
            process.wait(timeout=timeout)
            return process.returncode, "", ""
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out ({timeout}s): {' '.join(cmd)}")
        return -1, "", "Timeout"
    except Exception as e:
        print(f"❌ Command execution failed: {' '.join(cmd)}, Error: {str(e)}")
        return -2, "", str(e)

def check_git_installed():
    """Check if git is installed"""
    ret, _, _ = run_cmd(["git", "--version"])
    if ret != 0:
        print("❌ git is not installed. Please install git before running this script.")
        sys.exit(1)
    print("✅ git is installed")

def clean_triton_cache():
    """
    Clean Triton's cache directory (~/.triton/cache/*) with fault tolerance
    Handles cases where cache directory doesn't exist (no error thrown)
    """
    # Resolve ~ to current user's home directory (avoids path parsing issues)
    triton_cache_dir = Path("~/.triton/cache").expanduser()
    # Construct rm command (list format to prevent shell injection)
    rm_cmd = ["rm", "-rf", str(triton_cache_dir / "*")]

    print(f"\n🧹 Cleaning Triton cache: {triton_cache_dir}/*")
    ret, _, stderr = run_cmd(rm_cmd, capture_output=False)

    # Fault tolerance: Skip if cache directory doesn't exist
    if ret != 0:
        if "No such file or directory" in stderr:
            print(f"⚠️ Triton cache directory not found, skipping cleanup: {triton_cache_dir}")
        else:
            print(f"❌ Failed to clean Triton cache! Error: {stderr}")
            sys.exit(1)
    else:
        print(f"✅ Triton cache cleanup completed")

# ======================== Core Logic ========================
def main():
    # 1. Pre-check
    check_git_installed()
    llvm_local_repo = Path(LLVM_LOCAL_REPO_DIR)
    triton_local_repo = Path(TRITON_LOCAL_REPO_DIR)
    llvm_build_dir = (Path(LLVM_LOCAL_REPO_DIR) / "build").resolve()

    # 2. Clone repository
    if llvm_local_repo.exists():
        print(f"⚠️ Local repository directory {LLVM_LOCAL_REPO_DIR} already exists, skipping clone")
    else:
        ret, _, stderr = run_cmd(
            ["git", "clone", LLVM_REPO_URL],
            capture_output=False  # Print clone progress in real-time
        )
        if ret != 0:
            print(f"❌ Failed to clone repository: {stderr}")
            sys.exit(1)
        print("✅ Trion repository cloned successfully")

    if triton_local_repo.exists():
        print(f"⚠️ Local repository directory {TRITON_LOCAL_REPO_DIR} already exists, skipping clone")
    else:
        ret, _, stderr = run_cmd(
            ["git", "clone", TRITON_REPO_URL],
            capture_output=False  # Print clone progress in real-time
        )
        if ret != 0:
            print(f"❌ Failed to clone repository: {stderr}")
            sys.exit(1)
        print("✅ Trion repository cloned successfully")

    # 3. Add remote repository
    # Check if remote already exists first
    ret, stdout, _ = run_cmd(["git", "remote", "-v"], cwd=LLVM_LOCAL_REPO_DIR)
    if REMOTE_NAME in stdout:
        print(f"⚠️ Remote repository {REMOTE_NAME} already exists, skipping addition")
    else:
        ret, _, stderr = run_cmd(
            ["git", "remote", "add", REMOTE_NAME, LLVM_LOCAL_REPO_URL],
            cwd=LLVM_LOCAL_REPO_DIR
        )
        if ret != 0:
            print(f"❌ Failed to add remote llvm repository: {stderr}")
            sys.exit(1)
        print("✅ Remote llvm repository added successfully")

    ret, stdout, _ = run_cmd(["git", "remote", "-v"], cwd=TRITON_LOCAL_REPO_DIR)
    if REMOTE_NAME in stdout:
        print(f"⚠️ Remote repository {REMOTE_NAME} already exists, skipping addition")
    else:
        ret, _, stderr = run_cmd(
            ["git", "remote", "add", REMOTE_NAME, TRITON_LOCAL_REPO_URL],
            cwd=TRITON_LOCAL_REPO_DIR
        )
        if ret != 0:
            print(f"❌ Failed to add remote triton repository: {stderr}")
            sys.exit(1)
        print("✅ Remote triton repository added successfully")

    # 4. Fetch remote repository branch info (Mandatory before cherry-pick)
    ret, _, stderr = run_cmd(
        ["git", "fetch", REMOTE_NAME],
        cwd=LLVM_LOCAL_REPO_DIR,
        capture_output=False
    )
    if ret != 0:
        print(f"❌ Failed to fetch remote llvm repository info: {stderr}")
        sys.exit(1)
    print("✅ Remote llvm repository info fetched successfully")

    ret, _, stderr = run_cmd(
        ["git", "fetch", REMOTE_NAME],
        cwd=TRITON_LOCAL_REPO_DIR,
        capture_output=False
    )
    if ret != 0:
        print(f"❌ Failed to fetch remote triton repository info: {stderr}")
        sys.exit(1)
    print("✅ Remote triton repository info fetched successfully")

    # 5. Execute git cherry-pick
    llvm_hash_path = "./triton/cmake/llvm-hash.txt"
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
        cherry_out_cmd = ["git", "checkout", rev]
    ret, stdout, stderr = run_cmd(cherry_out_cmd, cwd=LLVM_LOCAL_REPO_DIR)
    if ret == 0:
        print("✅ LLVM checkout executed successfully")
    else:
        print(f"❌ LLVM checkout execution failed: {stderr}")
        sys.exit(1)

    if isinstance(LLVM_CHERRY_PICK_COMMITS, str):
        # Continuous commits: git cherry-pick start^..end
        cherry_pick_cmd = ["git", "cherry-pick", LLVM_CHERRY_PICK_COMMITS]
    else:
        # Non-continuous commits: git cherry-pick commit1 commit2 ...
        cherry_pick_cmd = ["git", "cherry-pick"] + LLVM_CHERRY_PICK_COMMITS

    ret, stdout, stderr = run_cmd(cherry_pick_cmd, cwd=LLVM_LOCAL_REPO_DIR)
    if ret == 0:
        print("✅ cherry-pick executed successfully")
    else:
        print(f"❌ cherry-pick execution failed: {stderr}")
        # Optional: Auto abort cherry-pick (Uncomment if needed)
        # run_cmd(["git", "cherry-pick", "--abort"], cwd=LOCAL_REPO_DIR)
        sys.exit(1)

    if isinstance(TRITON_CHERRY_PICK_COMMITS, str):
        # Continuous commits: git cherry-pick start^..end
        cherry_pick_cmd = ["git", "cherry-pick", TRITON_CHERRY_PICK_COMMITS]
    else:
        # Non-continuous commits: git cherry-pick commit1 commit2 ...
        cherry_pick_cmd = ["git", "cherry-pick"] + TRITON_CHERRY_PICK_COMMITS

    cherry_out_cmd = ["git", "checkout", "-b", "test"]

    ret, stdout, stderr = run_cmd(cherry_out_cmd, cwd=TRITON_LOCAL_REPO_DIR)
    if ret == 0:
        print("✅ triton checkout executed successfully")
    else:
        print(f"❌ triton checkout execution failed: {stderr}")
        sys.exit(1)

    ret, stdout, stderr = run_cmd(cherry_pick_cmd, cwd=TRITON_LOCAL_REPO_DIR)
    if ret == 0:
        print("✅ cherry-pick executed successfully")
    else:
        print(f"❌ cherry-pick execution failed: {stderr}")
        # Optional: Auto abort cherry-pick (Uncomment if needed)
        # run_cmd(["git", "cherry-pick", "--abort"], cwd=LOCAL_REPO_DIR)
        sys.exit(1)

    # 6. Build llvm and triton
    print("\n📁 Creating LLVM build subdirectory (explicit mkdir)")
    ret, _, stderr = run_cmd(["mkdir", "build"], cwd=LLVM_LOCAL_REPO_DIR)
    if ret != 0 and "File exists" not in stderr:  # Ignore if build dir exists
        print(f"❌ Failed to create build directory: {stderr}")
        sys.exit(1)

    ret, _, stderr = run_cmd(LLVM_CMAKE_CMD, cwd=llvm_build_dir, capture_output=False)
    if ret != 0:
        print(f"❌ CMake configuration failed: {stderr}")
        sys.exit(1)
    print("✅ LLVM CMake configuration completed successfully")

    print(f"\n🔨 Building LLVM with Ninja (using {os.cpu_count()} cores)")
    ninja_cmd = NINJA_BUILD_CMD.split()
    ret, _, stderr = run_cmd(ninja_cmd, cwd=llvm_build_dir, capture_output=False)
    if ret != 0:
        print(f"❌ LLVM build failed: {stderr}")
        sys.exit(1)
    print("✅ LLVM built successfully with Ninja")   

    os.environ["LLVM_DIR"] = str(llvm_build_dir)
    os.environ["LLVM_INCLUDE_DIRS"] = str(llvm_build_dir / "include")
    os.environ["LLVM_LIBRARY_DIR"] = str(llvm_build_dir / "lib")

    # Print env vars for verification
    print("\n🔧 LLVM Environment Variables (Verification):")
    print("-" * 60)
    print(f"   • LLVM_INCLUDE_DIRS:   {os.environ.get('LLVM_INCLUDE_DIRS')}")
    print(f"   • LLVM_LIBRARY_DIR:    {os.environ.get('LLVM_LIBRARY_DIR')}")
    print(f"   • LLVM_SYSPATH:        {os.environ.get('LLVM_DIR')}")
    print("-" * 60)

    # Run pip install -e . with custom env vars
    pip_cmd = ["pip3", "install", "-e", "."]
    ret, _, stderr = run_cmd(
        pip_cmd,
        cwd=TRITON_LOCAL_REPO_DIR,  # Run pip install from LLVM source dir (adjust if needed)
        capture_output=False  # Real-time pip install output
    )
    if ret != 0:
        print(f"❌ Pip install failed: {stderr}")
        sys.exit(1)
    print("✅ Pip install completed successfully (LLVM env vars applied)")

    # 8. Remove cache
    clean_triton_cache()

    # 9. Execute Shell test script
    if not Path(SHELL_SCRIPT_PATH).exists():
        print(f"❌ Shell script {SHELL_SCRIPT_PATH} does not exist")
        sys.exit(1)
    
    # Set execute permission for Shell script
    os.chmod(SHELL_SCRIPT_PATH, 0o755)
    shell_cmd = [SHELL_SCRIPT_NAME] + ["fp16"]
    ret, stdout, stderr = run_cmd(shell_cmd, cwd=TRITON_LOCAL_REPO_DIR, capture_output=False)
    if ret == 0:
        print("✅ Shell test script data type fp16 executed successfully")
    else:
        print(f"❌ Shell test script data type fp16 execution failed: {stderr}")
        sys.exit(1)

    shell_cmd = [SHELL_SCRIPT_NAME] + ["bf16"]
    ret, stdout, stderr = run_cmd(shell_cmd, cwd=TRITON_LOCAL_REPO_DIR, capture_output=False)
    if ret == 0:
        print("✅ Shell test script data type bf16 executed successfully")
    else:
        print(f"❌ Shell test script data type bf16 execution failed: {stderr}")
        sys.exit(1)

    print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()

