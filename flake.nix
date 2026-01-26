{
  description = "Aceteam Workflow Engine development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python312
            uv
            graphviz
          ];

          shellHook = ''
            echo "Workflow Engine dev environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
            echo "Graphviz: $(dot -V 2>&1)"
          '';
        };
      }
    );
}
