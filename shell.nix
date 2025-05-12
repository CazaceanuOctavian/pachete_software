{ pkgs ? import <nixpkgs> {} }: 

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python environment (using 3.12 from the second file)
    (python312.withPackages (ps: with ps; [
      # Basic data science packages
      numpy
      pandas
      matplotlib
      scipy
      scikit-learn
      seaborn
      
      # Machine learning packages
      xgboost
      catboost
      
      # Streamlit and dependencies
      streamlit
      
      # Dev tools and utilities
      ipython
      jupyter
      notebook
      black
      pylint
      pytest
      requests
      pillow
      pip
    ]))
    
    # System dependencies
    gcc
    openssl
  ];
  
  shellHook = ''
    # Environment setup from second file
    export PIP_PREFIX="$(pwd)/_build/pip_packages"
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python312.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
    
    # Welcome message from first file
    echo "Python development environment with Streamlit and ML tools activated!"
    echo "Run 'streamlit hello' to verify the Streamlit installation"
    echo "Run 'streamlit run your_app.py' to start your Streamlit app"
    echo "Jupyter is also available - run 'jupyter notebook' to start"
  '';
}
