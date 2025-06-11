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
  };

  # Environment variables for integration
  env = {
    PYTHONPATH = lib.makeSearchPath "lib/python3.11/site-packages"	[
      config.deps.artist
    ];
    # JAX will auto-detect best backend (Metal on macOS)
  };
}