
ur3robotiq140.ttm and ur3robotiq85.ttm need to be fixed.
You need to save the model as an .xml file and make sur the following lines are inside:

    <constraints>
        <x>true</x>
        <y>true</y>
        <z>true</z>
        <alpha_beta>true</alpha_beta>
        <gamma>true</gamma>
    </constraints>
