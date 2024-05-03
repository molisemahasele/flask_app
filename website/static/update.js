function updateNote(noteId, updatedContent) {
    fetch("/update-note", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ noteId, updatedContent }),
    }).then((_res) => {
        // Redirect to the homepage or a success page after updating the note
        window.location.href = "/";
    }).catch((error) => {
        console.error('Error updating note:', error);
        // Handle errors, e.g., display an error message to the user
    });
}
