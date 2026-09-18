find_program(PATCH_EXECUTABLE patch REQUIRED)
execute_process(COMMAND "${PATCH_EXECUTABLE}" --batch -p1 -i "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Could not apply the pinned TTS CPU kernel patch")
endif()
