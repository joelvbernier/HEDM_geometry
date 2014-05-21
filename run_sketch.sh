WORK_DIR=$1

cd ${WORK_DIR}

sketch -o ${WORK_DIR}/new_geometry.sk.out ${WORK_DIR}/new_geometry.sk
latex ${WORK_DIR}/new_geometry.tex
dvips -o ${WORK_DIR}/new_geometry.ps ${WORK_DIR}/new_geometry.dvi
dvipdf ${WORK_DIR}/new_geometry.dvi ${WORK_DIR}/new_geometry.pdf
