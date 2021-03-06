NODE_LIST = [
    'set', 'str', 'bool', 'CompilationUnit', 'Import', 'Documented', 'Declaration', 'TypeDeclaration',
    'PackageDeclaration', 'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration',
    'AnnotationDeclaration', 'Type', 'BasicType', 'ReferenceType', 'TypeArgument',
    'TypeParameter', 'Annotation', 'ElementValuePair', 'ElementArrayValue', 'Member',
    'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration', 'ConstantDeclaration',
    'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration', 'VariableDeclarator',
    'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement', 'WhileStatement',
    'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement', 'ContinueStatement',
    'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement', 'SwitchStatement',
    'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause', 'CatchClauseParameter',
    'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression', 'Assignment', 'TernaryExpression',
    'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression', 'Primary', 'Literal', 'This',
    'MemberReference', 'Invocation', 'ExplicitConstructorInvocation', 'SuperConstructorInvocation',
    'MethodInvocation', 'SuperMethodInvocation', 'SuperMemberReference', 'ArraySelector', 'ClassReference',
    'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator', 'InnerClassCreator', 'EnumBody',
    'EnumConstantDeclaration', 'AnnotationMethod']

NODE_MAP = {x: i for (i, x) in enumerate(NODE_LIST)}
