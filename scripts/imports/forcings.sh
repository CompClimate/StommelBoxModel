s_forcings=("sinusoidal_stationary" "sinusoidal_nonstationary" "linear_symmetric" "sinusoidal_stationary" "sinusoidal_nonstationary" "linear_symmetric")
t_forcings=(null null null "sinusoidal_stationary" "sinusoidal_nonstationary" "linear_symmetric")

get_t_forcing () {
	t_forcing=""
	if [[ "${t_forcings[$1]}" == null ]];
	then
		t_forcing="~t_forcing"
	else
		t_forcing="t_forcing=${t_forcings[$1]}"
	fi
	echo $t_forcing
}

function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}
