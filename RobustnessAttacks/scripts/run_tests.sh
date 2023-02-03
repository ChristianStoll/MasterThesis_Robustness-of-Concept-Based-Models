#!/usr/bin/env bash
python3 workstation_attacks.py --task attack_img_XtoC --batch_size 32
python3 workstation_attacks.py --task attack_concepts_CtoY --batch_size 32
python3 workstation_attacks.py --task test_scouter_chkpts --batch_size 32
python3 workstation_attacks.py --task attack_e2e_cub_black --batch_size 32
python3 workstation_attacks.py --task attack_scouter_variations --batch_size 64