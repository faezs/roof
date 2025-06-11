# Heliostat control system with MuJoCo-MJX, ARTIST, and SBV verification
{
  config,
  lib,
  dream2nix,
  ...
}: let
  pyproject = lib.importTOML (config.mkDerivation.src + /pyproject.toml);
in {
  imports = [
    dream2nix.modules.dream2nix.pip
  ];

  deps = {nixpkgs, ...}: {
    python = nixpkgs.python3;
    inherit (nixpkgs) stdenv;
    inherit (nixpkgs) runCommand;
    inherit (nixpkgs) fetchFromGitHub;
    inherit (nixpkgs) fetchPypi;
    inherit (nixpkgs) hatch;
    hatchling = nixpkgs.python3.pkgs.hatchling;
    
    # Get the ARTIST package from our existing flake input
    artist = config.deps.artistPackage;
  };

  inherit (pyproject.project) name version;

  mkDerivation = {
    src = lib.cleanSourceWith {
      src = lib.cleanSource ./.;
      filter = name: type:
        !(builtins.any (x: x) [
          (lib.hasSuffix ".nix" name)
          (lib.hasPrefix "." (builtins.baseNameOf name))
          (lib.hasSuffix "flake.lock" name)
          (lib.hasSuffix "__pycache__" name)
          (lib.hasSuffix ".pyc" name)
        ]);
    };
  };

  buildPythonPackage = {
    pyproject = true;
    build-system = [ config.deps.python.pkgs.hatchling ];
    pythonImportsCheck = [
      "mujoco_heliostat_sim"
      "mjx_autodiff_control" 
      "mjx_artist_integration"
    ];
  };

  pip = {
    requirementsList =
      pyproject.build-system.requires or []
      ++ pyproject.project.dependencies or []
      ++ pyproject.project.optional-dependencies.dev or []
      ++ pyproject.project.optional-dependencies.simulation or [];
    
    flattenDependencies = true;
    
    # Override packages that need special handling
    overrides = {
      # Use PyTorch from nixpkgs for better compatibility
      torch = let
        nixpkgsTorch = config.deps.python.pkgs.torch;
        torchWheel = config.deps.runCommand "torch-wheel" {} ''
          file="$(ls "${nixpkgsTorch.dist}")"
          mkdir "$out"
          cp "${nixpkgsTorch.dist}/$file" "$out/$file"
        '';
      in {
        mkDerivation.src = torchWheel;
        mkDerivation.prePhases = ["selectWheelFile"];
        env.selectWheelFile = ''
          export src="$src/$(ls $src)"
        '';
        mkDerivation.buildInputs = [
          nixpkgsTorch.inputDerivation
        ];
        buildPythonPackage = {
          format = "wheel";
          pyproject = null;
        };
      };

      # Use torchvision from nixpkgs
      torchvision = let
        nixpkgsTorchvision = config.deps.python.pkgs.torchvision;
        torchvisionWheel = config.deps.runCommand "torchvision-wheel" {} ''
          file="$(ls "${nixpkgsTorchvision.dist}")"
          mkdir "$out"
          cp "${nixpkgsTorchvision.dist}/$file" "$out/$file"
        '';
      in {
        mkDerivation.src = torchvisionWheel;
        mkDerivation.prePhases = ["selectWheelFile"];
        env.selectWheelFile = ''
          export src="$src/$(ls $src)"
        '';
        mkDerivation.buildInputs = [
          nixpkgsTorchvision.inputDerivation
        ];
        buildPythonPackage = {
          format = "wheel";
          pyproject = null;
        };
      };

      # Use JAX from nixpkgs for MPS/CUDA support
      jax = let
        nixpkgsJax = config.deps.python.pkgs.jax;
        jaxWheel = config.deps.runCommand "jax-wheel" {} ''
          file="$(ls "${nixpkgsJax.dist}")"
          mkdir "$out" 
          cp "${nixpkgsJax.dist}/$file" "$out/$file"
        '';
      in {
        mkDerivation.src = jaxWheel;
        mkDerivation.prePhases = ["selectWheelFile"];
        env.selectWheelFile = ''
          export src="$src/$(ls $src)"
        '';
        mkDerivation.buildInputs = [
          nixpkgsJax.inputDerivation
        ];
        buildPythonPackage = {
          format = "wheel";
          pyproject = null;
        };
      };

      # Use our built ARTIST package
      artist = lib.mkIf (config.deps.artist != null) {
        mkDerivation.src = config.deps.artist;
	        mkDerivation.prePhases = ["selectWheelFile"];
        env.selectWheelFile = ''
          export src="$src/$(ls $src)"
        '';
        buildPythonPackage = {
          format = "wheel";
          pyproject = null;
        };
      };
    };
  };

  # Environment variables for integration
  env = {
    PYTHONPATH = lib.makeSearchPath "lib/python3.11/site-packages" [
      config.deps.artist
    ];
  };
}