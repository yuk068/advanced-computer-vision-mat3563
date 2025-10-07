import os

def print_directory_tree(start_path='.', file_limit=3, ignored_dirs=None):
    """
    Prints the directory tree of a given path in ASCII format,
    focusing on folders, but showing a limited list of files.

    :param start_path: The path to start the traversal from (default is current directory).
    :param file_limit: Maximum number of files to show (if exceeded, '...' is added).
    :param ignored_dirs: A list of directory names to ignore during traversal.
    """
    if ignored_dirs is None:
        ignored_dirs = []
    
    # Prepend the ignored_dirs with a default set for clean output
    default_ignored = ['.git', '__pycache__', '.ipynb_checkpoints', 'node_modules', '.idea']
    # Use a set to ensure unique ignored directories
    ignored_dirs = list(set(ignored_dirs + default_ignored))

    print(f"** Directory Tree for: {os.path.abspath(start_path)} **\n")

    # The actual recursive function
    def tree(root, prefix=''):
        # Get all entries (directories and files)
        try:
            contents = os.listdir(root)
        except OSError as e:
            print(f"{prefix}└── [ERROR: Cannot access {root} - {e}]")
            return

        # Separate directories and files
        dirs = sorted([d for d in contents if os.path.isdir(os.path.join(root, d)) and d not in ignored_dirs])
        files = sorted([f for f in contents if os.path.isfile(os.path.join(root, f))])
        
        # --- File Display Logic ---
        
        # 1. Prepare the list of files to display
        files_to_display = files[:file_limit]
        
        # 2. Check if the '...' indicator is needed
        show_ellipsis = (len(files) > file_limit)
        
        # 3. Combine entries: Folders first, then the files/ellipsis
        entries = dirs + files_to_display
        
        # 4. If we are showing the ellipsis, it counts as one "entry" at the end
        if show_ellipsis:
            entries.append("...")

        # Total number of entries to display
        n_entries = len(entries)
        
        for i, entry in enumerate(entries):
            # Is this the last entry in the current directory listing?
            is_last = (i == n_entries - 1)
            
            # Choose the appropriate connectors
            connector = '└── ' if is_last else '├── '
            
            # Print the current entry
            print(f"{prefix}{connector}{entry}")
            
            # If it's a directory, recurse
            if entry in dirs:
                full_path = os.path.join(root, entry)
                # Update the prefix for the next level
                # '    ' if last, '|   ' otherwise
                next_prefix = prefix + ('    ' if is_last else '│   ')
                tree(full_path, next_prefix)

    # Start the traversal
    tree(start_path)

# --- Example Usage ---
if __name__ == '__main__':
    # You would typically run this in a directory with subfolders
    # and files to see the file limiting in action.
    print("--- Directory Tree with File Limiting (Max 3 Files) ---")
    print_directory_tree(start_path='.', file_limit=3)
    
    print("\n" + "="*50 + "\n")